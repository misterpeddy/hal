import gc
import os
import warnings
from pathlib import Path

import joblib
import librosa as rosa
import librosa.display
import madmom as mm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal as signal
import sklearn.cluster
import torch as torch
import torch.nn.functional as F

from hal_project.hal import analyzers

SMF = 1

class Scene:
  def __init__(self, generator, track):
    self.generator = generator
    self.track = track
    self.n_frames = track.audio.shape[0]
    low_onsets = analyzers.onsets(track, fmax=150)
    high_onsets = analyzers.onsets(track, fmin=500)
    self.latents = get_latents(generator, track, low_onsets, high_onsets)
    self.noise = get_noise(generator, low_onsets, high_onsets)

  def render(self):
    return _render(self)

def get_latents(generator, track, low_onsets, high_onsets):
  chroma = analyzers.chroma(track)
  note_latents = generator.generate_latents(12)
  chroma_latents = chroma_weight_latents(chroma, note_latents)
  latents = gaussian_filter(chroma_latents, 4)

  latents = high_onsets * note_latents[[-4]] + (1 - high_onsets) * latents
  latents = low_onsets * note_latents[[-7]] + (1 - low_onsets) * latents

  latents = gaussian_filter(latents, 2, causal=0.2)

  return latents

def get_noise(generator, low_onsets, high_onsets):
  noise = []
  range_min, range_max, exponent = generator.get_noise_range()
  for scale in range(range_min, range_max):
    h = 2 ** exponent(scale)
    w = 2 ** exponent(scale)

    noise.append(get_noise_at_scale(height=h, width=w, scale=scale - range_min, num_scales=range_max - range_min, onsets_list=[low_onsets, high_onsets], n_frames=n_frames))

    if noise[-1] is not None:
      print(list(noise[-1].shape), f"amplitude={noise[-1].std()}")
    gc.collect()
    torch.cuda.empty_cache()
    print()
  return noise

def chroma_weight_latents(chroma, latents):
  """Creates chromagram weighted latent sequence

  Args:
      chroma (th.tensor): Chromagram
      latents (th.tensor): Latents (must have same number as number of notes in chromagram)

  Returns:
      th.tensor: Chromagram weighted latent sequence
  """
  base_latents = (chroma[..., None, None] * latents[None, ...]).sum(1)
  return base_latents

def gaussian_filter(x, sigma, causal=None):
  """Smooth tensors along time (first) axis with gaussian kernel.

  Args:
      x (th.tensor): Tensor to be smoothed
      sigma (float): Standard deviation for gaussian kernel (higher value gives smoother result)
      causal (float, optional): Factor to multiply right side of gaussian kernel with. Lower value decreases effect of "future" values. Defaults to None.

  Returns:
      th.tensor: Smoothed tensor
  """
  dim = len(x.shape)
  n_frames = x.shape[0]
  while len(x.shape) < 3:
    x = x[:, None]

  radius = min(int(sigma * 4 * SMF), 3 * len(x))
  channels = x.shape[1]

  kernel = torch.arange(-radius, radius + 1, dtype=th.float32, device=x.device)
  kernel = torch.exp(-0.5 / sigma ** 2 * kernel ** 2)
  if causal is not None:
    kernel[radius + 1 :] *= 0 if not isinstance(causal, float) else causal
  kernel = kernel / kernel.sum()
  kernel = kernel.view(1, 1, len(kernel)).repeat(channels, 1, 1)

  if dim == 4:
    t, c, h, w = x.shape
    x = x.view(t, c, h * w)
  x = x.transpose(0, 2)

  if radius > n_frames:  # prevent padding errors on short sequences
    x = F.pad(x, (n_frames, n_frames), mode="circular")
    print(
      f"WARNING: Gaussian filter radius ({int(sigma * 4 * SMF)}) is larger than number of frames ({n_frames}).\n\t Filter size has been lowered to ({radius}). You might want to consider lowering sigma ({sigma})."
    )
    x = F.pad(x, (radius - n_frames, radius - n_frames), mode="constant")
  else:
    x = F.pad(x, (radius, radius), mode="circular")

  x = F.conv1d(x, weight=kernel, groups=channels)

  x = x.transpose(0, 2)
  if dim == 4:
    x = x.view(t, c, h, w)

  if len(x.shape) > dim:
    x = x.squeeze()

  return x

def get_noise_at_scale(height, width, scale, num_scales, onsets_list, n_frames):
  if width > 256:
    return None

  lo_onsets = onsets[0][:, None, None, None].cuda()
  hi_onsets = onsets[1][:, None, None, None].cuda()

  noise_noisy = gaussian_filter(torch.randn((n_frames, 1, height, width), device="cuda"), 5)
  noise = gaussian_filter(torch.randn((n_frames, 1, height, width), device="cuda"), 128)

  if width < 128:
    noise = lo_onsets * noise_noisy + (1 - lo_onsets) * noise
  if width > 32:
    noise = hi_onsets * noise_noisy + (1 - hi_onsets) * noise

  noise /= noise.std() * 2.5

  return noise.cpu()

import queue
from threading import Thread

import ffmpeg
import numpy as np
import PIL.Image
import torch as th
from tqdm import tqdm

def _render(
  scene,
  generator,
  offset,
  duration,
  batch_size=8,
  out_size=512,
  output_file="/tmp/output.mp4",
  truncation=1.0,
  bends=[],
  rewrites={},
  randomize_noise=False,
  ffmpeg_preset="slow",
):
  latents = scene.latents
  noise = scene.noise
  audio_file = scene.track.path

  torch.set_grad_enabled(False)
  torch.backends.cudnn.benchmark = True

  split_queue = queue.Queue()
  render_queue = queue.Queue()

  # postprocesses batched torch tensors to individual RGB numpy arrays
  def split_batches(jobs_in, jobs_out):
    while True:
      try:
        imgs = jobs_in.get(timeout=5)
      except queue.Empty:
        return
      imgs = (imgs.clamp_(-1, 1) + 1) * 127.5
      imgs = imgs.permute(0, 2, 3, 1)
      for img in imgs:
        jobs_out.put(img.cpu().numpy().astype(np.uint8))
      jobs_in.task_done()

  # start background ffmpeg process that listens on stdin for frame data
  if out_size == 512:
    output_size = "512x512"
  elif out_size == 1024:
    output_size = "1024x1024"
  elif out_size == 1920:
    output_size = "1920x1080"
  elif out_size == 1080:
    output_size = "1080x1920"
  else:
    raise Exception("The only output sizes currently supported are: 512, 1024, 1080, or 1920")

  audio = ffmpeg.input(audio_file, ss=offset, t=duration, guess_layout_max=0)
  video = (
    ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=len(latents) / duration, s=output_size)
    .output(
      audio,
      output_file,
      framerate=len(latents) / duration,
      vcodec="libx264",
      pix_fmt="yuv420p",
      preset=ffmpeg_preset,
      audio_bitrate="320K",
      ac=2,
      v="warning",
    )
    .global_args("-hide_banner")
    .overwrite_output()
    .run_async(pipe_stdin=True)
  )

  # writes numpy frames to ffmpeg stdin as raw rgb24 bytes
  def make_video(jobs_in):
    w, h = [int(dim) for dim in output_size.split("x")]
    for _ in tqdm(range(len(latents)), position=0, leave=True, ncols=80):
        img = jobs_in.get(timeout=5)
        if img.shape[1] == 2048:
            img = img[:, 112:-112, :]
            im = PIL.Image.fromarray(img)
            img = np.array(im.resize((1920, 1080), PIL.Image.BILINEAR))
        elif img.shape[0] == 2048:
            img = img[112:-112, :, :]
            im = PIL.Image.fromarray(img)
            img = np.array(im.resize((1080, 1920), PIL.Image.BILINEAR))
        assert (
            img.shape[1] == w and img.shape[0] == h
        ), f"""generator's output image size does not match specified output size: \n
            got: {img.shape[1]}x{img.shape[0]}\t\tshould be {output_size}"""
        video.stdin.write(img.tobytes())
        jobs_in.task_done()
    video.stdin.close()
    video.wait()

  splitter = Thread(target=split_batches, args=(split_queue, render_queue))
  splitter.daemon = True
  renderer = Thread(target=make_video, args=(render_queue,))
  renderer.daemon = True

  # make all data that needs to be loaded to the GPU float, contiguous, and pinned
  # the entire process is severly memory-transfer bound, but at least this might help a little
  latents = latents.float().contiguous().pin_memory()

  for ni, noise_scale in enumerate(noise):
      noise[ni] = noise_scale.float().contiguous().pin_memory() if noise_scale is not None else None

  param_dict = dict(generator.named_parameters())
  original_weights = {}
  for param, (rewrite, modulation) in rewrites.items():
      rewrites[param] = [rewrite, modulation.float().contiguous().pin_memory()]
      original_weights[param] = param_dict[param].copy().cpu().float().contiguous().pin_memory()

  for bend in bends:
      if "modulation" in bend:
          bend["modulation"] = bend["modulation"].float().contiguous().pin_memory()

  if not isinstance(truncation, float):
      truncation = truncation.float().contiguous().pin_memory()

  for n in range(0, len(latents), batch_size):
      # load batches of data onto the GPU
      latent_batch = latents[n : n + batch_size].cuda(non_blocking=True)

      noise_batch = []
      for noise_scale in noise:
          if noise_scale is not None:
              noise_batch.append(noise_scale[n : n + batch_size].cuda(non_blocking=True))
          else:
              noise_batch.append(None)

      bend_batch = []
      if bends is not None:
          for bend in bends:
              if "modulation" in bend:
                  transform = bend["transform"](bend["modulation"][n : n + batch_size].cuda(non_blocking=True))
                  bend_batch.append({"layer": bend["layer"], "transform": transform})
              else:
                  bend_batch.append({"layer": bend["layer"], "transform": bend["transform"]})

      for param, (rewrite, modulation) in rewrites.items():
          transform = rewrite(modulation[n : n + batch_size])
          rewritten_weight = transform(original_weights[param]).cuda(non_blocking=True)
          param_attrs = param.split(".")
          mod = generator
          for attr in param_attrs[:-1]:
              mod = getattr(mod, attr)
          setattr(mod, param_attrs[-1], th.nn.Parameter(rewritten_weight))

      if not isinstance(truncation, float):
          truncation_batch = truncation[n : n + batch_size].cuda(non_blocking=True)
      else:
          truncation_batch = truncation

      # forward through the generator
      outputs, _ = generator(
          styles=latent_batch,
          noise=noise_batch,
          truncation=truncation_batch,
          transform_dict_list=bend_batch,
          randomize_noise=randomize_noise,
          input_is_latent=True,
      )

      # send output to be split into frames and rendered one by one
      split_queue.put(outputs)

      if n == 0:
          splitter.start()
          renderer.start()

  splitter.join()
  renderer.join()


def write_video(arr, output_file, fps):
  print(f"writing {arr.shape[0]} frames...")

  output_size = "x".join(reversed([str(s) for s in arr.shape[1:-1]]))

  ffmpeg_proc = (
    ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=fps, s=output_size)
    .output(output_file, framerate=fps, vcodec="libx264", preset="slow", v="warning")
    .global_args("-benchmark", "-stats", "-hide_banner")
    .overwrite_output()
    .run_async(pipe_stdin=True)
  )

  for frame in arr:
    ffmpeg_proc.stdin.write(frame.astype(np.uint8).tobytes())

  ffmpeg_proc.stdin.close()
  ffmpeg_proc.wait()
