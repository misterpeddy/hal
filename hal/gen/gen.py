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

from hal_project.hal.audio import analyzers
from hal_project.hal import io_utils 

FRAME_RATE = 24

class Scene:
  def __init__(self, model, track):
    self.model = model
    self.track = track
    self.n_frames = FRAME_RATE * int(track.audio.shape[0] / track.sr)
    self.latents = self.get_latents(model, track)
    self.noise = self.get_noise(model, track)

  def render(self):
    return _render(self)

  def get_latents(self, model, track):
    note_latents = model._generate_latents(12).cpu()
    chroma = analyzers.chroma(track)
    chroma_latents = chroma_weight_latents(chroma, note_latents)

    self.note_latents = note_latents
    self.chroma = chroma
    self.chroma_latents = chroma_latents
    
    latents = analyzers.gaussian_filter(chroma_latents, 4)

    high_onsets = track._high_onsets[:, None, None]
    low_onsets = track._low_onsets[:, None, None]

    latents = (high_onsets * note_latents[[-4]]) + ((1 - high_onsets) * latents)
    latents = low_onsets * note_latents[[-7]] + (1 - low_onsets) * latents

    latents = analyzers.gaussian_filter(latents, 2, causal=0.2)

    return latents

  def get_noise(self, generator, track, amplification=4):
    
    low_onsets = track._low_onsets.cuda()[:, None, None, None]
    high_onsets = track._high_onsets.cuda()[:, None, None, None]
    assert low_onsets.shape[0] == high_onsets.shape[0]
    n_frames = low_onsets.shape[0]

    noises = []
    range_min, range_max, exponent = generator.get_noise_range()
    
    for scale in range(range_min, range_max):
      height = 2 ** exponent(scale)
      width = 2 ** exponent(scale)

      if width > 256:
        noises.append(None)
        continue

      noise_noisy = analyzers.gaussian_filter(torch.randn((n_frames, 1, height, width), device="cuda"), 5)
      noise = analyzers.gaussian_filter(torch.randn((n_frames, 1, height, width), device="cuda"), 64)

      if width < 128:
        noise = low_onsets * noise_noisy + (1 - low_onsets) * noise
      if width > 32:
        noise = high_onsets * noise_noisy + (1 - high_onsets) * noise

      noise /= noise.std()
      noise *= amplification
      noises.append(noise.cpu())

      print(list(noise[-1].shape), f"amplitude={noise[-1].std()}")
      
      gc.collect()
      torch.cuda.empty_cache()
      print()
    
    return noises

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

def get_noise_at_scale(height, width, n_frames, track):

  if width > 256:
    return None

  low_onsets = track._low_onsets.cuda()[:, None, None, None]
  high_onsets = track._high_onsets.cuda()[:, None, None, None]

  noise_noisy = analyzers.gaussian_filter(torch.randn((n_frames, 1, height, width), device="cuda"), 5)
  noise = analyzers.gaussian_filter(torch.randn((n_frames, 1, height, width), device="cuda"), 128)

  if width < 128:
    noise = low_onsets * noise_noisy + (1 - low_onsets) * noise
  if width > 32:
    noise = high_onsets * noise_noisy + (1 - high_onsets) * noise

  noise /= noise.std()

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
  batch_size=32,
  output_file=None,
  truncation=1.0,
  randomize_noise=False,
  ffmpeg_preset="slow",
):
  generator = scene.model.g_ema
  duration = scene.n_frames / 24
  latents = scene.latents
  noise = scene.noise
  audio_file = scene.track.path
  offset = 0
  out_size = scene.model.output_size

  if output_file is None:
      output_file = os.path.join(
      "/home/peddy_google_com/gen_dir/video",
      io_utils.name_from_path(audio_file))
      output_file = io_utils.next_dir(output_file)
      output_file = f'{output_file}.mp4'


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
  if out_size == 256:
    output_size = "256x256"
  elif out_size == 512:
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
  return output_file

def _complete_render(
  scene,
  batch_size=8,
  out_size=512,
  output_file=None,
  truncation=1.0,
  bends=[],
  rewrites={},
  randomize_noise=False,
  ffmpeg_preset="slow",
):
  generator = scene.model.g_ema
  duration = scene.n_frames / 24
  latents = scene.latents
  noise = scene.noise
  audio_file = scene.track.path
  offset = 0
  if output_file is None:
      output_file = os.path.join(
      "/home/peddy_google_com/gen_dir/video",
      io_utils.name_from_path(audio_file))
      output_file = io_utils.next_dir(output_file)
      output_file = f'{output_file}.mp4'


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
  if out_size == 256:
    output_size = "256x256"
  elif out_size == 512:
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
          noise=None,
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
  return output_file


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
