import gc
import os
import warnings
from pathlib import Path

import joblib
import librosa as rosa
import librosa.display
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal as signal
import sklearn.cluster
import torch as torch

from hal.audio import analyzers
from hal import io_utils

import hal

VID_FRAME_RATE = 24

def slerp(start, end, weight):
  """Interpolation along geodesic of n-dimensional unit sphere
  Args:
      val (float): Value between 0 and 1 representing fraction of interpolation completed
      low (float): Starting value
      high (float): Ending value
  Returns:
      float: Interpolated value
  """
  omega = np.arccos(np.clip(np.dot(start / np.linalg.norm(start), end / np.linalg.norm(end)), -1, 1))
  so = np.sin(omega)
  if so == 0:
      return (1.0 - weight) * start + weight * end  # L'Hopital's rule/LERP
  return np.sin((1.0 - weight) * omega) / so * start + np.sin(weight * omega) / so * end

def lerp(start, end, weights):
  """Returns a linear interploation between a and b, weighted by weights."""
  return (weights * start) + ((1 - weights) * end)

def add_movement(lerp, track):
  """Adds random sinusoidal movement in each of the latent dims"""
  shape = lerp.shape

  f_0 = 10
  a_0 = 0.5
  a_1 = 0.1
  f = f_0 * torch.rand(shape[1:])
  a = a_0 * torch.rand(shape[1:]) + a_1

  dx = 0.001
  x = torch.arange(0, dx * shape[0], dx)
  x = x.reshape(1, 1, x.shape[0])
  y = a.unsqueeze(2) * torch.sin(f.unsqueeze(2) * x)
  y = y.permute(2, 0, 1)

  moved_lerp = y * lerp
  scale = (track._low_onsets ) * (10 ** 0.1)
  scale = scale.reshape(scale.shape[0], 1, 1)
  scale = scale * torch.ones_like(lerp)
  scaled_lerp = scale * moved_lerp
  return scaled_lerp

def get_interp_onsets_latents_with_movement(model, track):
  lerp_latents = get_interp_onsets_latents(model, track)
  lerp_with_movement = add_movement(lerp_latents, track)
  return lerp_with_movement

def get_interp_onsets_latents(model, track):
  start, end = model._generate_latents(2).cpu()
  norm_low_onsets = track._low_onsets / track._low_onsets.sum()
  cum_onsets = torch.cumsum(norm_low_onsets, 0)
  weights = cum_onsets.unsqueeze(1).unsqueeze(2)
  return lerp(start, end, weights)

def chroma_latents(model, track):
  note_latents = model._generate_latents(12).cpu()
  chroma = analyzers.chroma(track)
  chroma_latents = chroma_weight_latents(chroma, note_latents)

  smooth_latents = analyzers.gaussian_filter(chroma_latents, 4)

  high_onsets = track._high_onsets[:, None, None]
  low_onsets = track._low_onsets[:, None, None]

  smooth_latents = (high_onsets * note_latents[[-4]]) + ((1 - high_onsets) * smooth_latents)
  smooth_latents = low_onsets * note_latents[[-7]] + (1 - low_onsets) * smooth_latents

  smooth_latents = analyzers.gaussian_filter(smooth_latents, 2, causal=0.2)

  return smooth_latents

def noise(generator, track, amplification=4):

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

    noise_noisy = analyzers.gaussian_filter(torch.randn((n_frames, 1, height, width), device="cuda"), 4)
    noise = analyzers.gaussian_filter(torch.randn((n_frames, 1, height, width), device="cuda"), 32)

    if width < 128:
      noise = low_onsets * noise_noisy + (1 - low_onsets) * noise
    if width > 32:
      noise = high_onsets * noise_noisy + (1 - high_onsets) * noise

    noise /= noise.std()
    noise *= amplification
    noises.append(noise.cpu())

    #print(list(noise[-1].shape), f"amplitude={noise[-1].std()}")

    gc.collect()
    torch.cuda.empty_cache()

  return noises

def empty_noise(generator):
  noises = []
  range_min, range_max, exponent = generator.get_noise_range()

  for scale in range(range_min, range_max):
    noises.append(None)

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

def render(
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
    scene.gen_dir,
    io_utils.name_from_path(audio_file))
    output_file = io_utils.next_dir(output_file)
    output_file = f'{output_file}.mp4'


  torch.set_grad_enabled(False)
  torch.backends.cudnn.benchmark = True

  split_queue = queue.Queue()
  render_queue = queue.Queue()

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

  splitter = Thread(target=split_batches, args=(split_queue, render_queue))
  splitter.daemon = True
  renderer = Thread(target=make_video, args=(render_queue, video, output_size, len(latents)))
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


# writes numpy frames to ffmpeg stdin as raw rgb24 bytes

def make_video(jobs_in, ffmpeg_proc, output_size, n_frames):
  w, h = [int(dim) for dim in output_size.split("x")]
  for _ in tqdm(range(n_frames), position=0, leave=True, ncols=80):
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
    ffmpeg_proc.stdin.write(img.tobytes())
    jobs_in.task_done()
  ffmpeg_proc.stdin.close()
  ffmpeg_proc.wait()
