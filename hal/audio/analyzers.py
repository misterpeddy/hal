import torch
import scipy.signal as signal
import librosa as rosa
import numpy as np

import torch.nn.functional as F

from hal_project.hal.gen.gen import FRAME_RATE

def onsets(track, fmin=80, fmax=2000, power=1):
  audio = rosa.effects.percussive(y=track.audio, margin=8)
  onset_frames = rosa.onset.onset_strength(audio, sr=track.sr, fmin=fmin, fmax=fmax)
  n_frames = int(FRAME_RATE * audio.shape[0] / track.sr)
  onset_frames = np.clip(signal.resample(onset_frames, n_frames), onset_frames.min(), onset_frames.max())
  onset_frames = torch.from_numpy(onset_frames).float()
  onset_frames = gaussian_filter(onset_frames, 1)
  onset_frames = percentile_clip(onset_frames)
  onset_frames = onset_frames ** power
  return onset_frames

def chroma(track, margin=16, notes=12):
    """Creates chromagram for the harmonic component of the audio

    Args:
        track (Track): Track to compute chromagram for.
        n_frames (int): Total number of frames to resample envelope to
        margin (int, optional): For harmonic source separation, higher values create more extreme separations. Defaults to 16.
        notes (int, optional): Number of notes to use in output chromagram (e.g. 5 for pentatonic scale, 7 for standard western scales). Defaults to 12.

    Returns:
        th.tensor, shape=(n_frames, 12): Chromagram
    """
    y_harm = rosa.effects.harmonic(y=track.audio, margin=margin)
    chroma = rosa.feature.chroma_cens(y_harm, sr=track.sr).T
    n_frames = int(FRAME_RATE * track.audio.shape[0] / track.sr)
    chroma = signal.resample(chroma, n_frames)
    notes_indices = np.argsort(np.median(chroma, axis=0))[:notes]
    chroma = chroma[:, notes_indices]
    chroma = torch.from_numpy(chroma / chroma.sum(1)[:, None]).float()
    return chroma

def percentile(signal, p):
    """Calculate percentile of signal

    Args:
        signal (np.array/th.tensor): Signal to normalize
        p (int): [0-100]. Percentile to find

    Returns:
        int: Percentile signal value
    """
    k = 1 + round(0.01 * float(p) * (signal.numel() - 1))
    return signal.view(-1).kthvalue(k).values.item()

def percentile_clip(signal, p=100):
  """Normalize signal between 0 and 1, clipping peak values above given percentile

  Args:
      signal (torch.tensor): Signal to normalize
      p (int): [0-100]. Percentile to clip to

  Returns:
      th.tensor: Clipped signal
  """
  locs = torch.arange(0, signal.shape[0])
  peaks = torch.ones(signal.shape, dtype=bool)
  main = signal.take(locs)

  plus = signal.take((locs + 1).clamp(0, signal.shape[0] - 1))
  minus = signal.take((locs - 1).clamp(0, signal.shape[0] - 1))
  peaks &= torch.gt(main, plus)
  peaks &= torch.gt(main, minus)

  signal = signal.clamp(0, percentile(signal[peaks], p))
  signal /= signal.max()
  return signal

def chromagram(track, harmonic_separation_margin=4):
  harmonic = rosa.effects.harmonic(track.audio, margin=harmonic_separation_margin)
  ch = rosa.feature.chroma_cens(harmonic, track.sr)
  return ch

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

  SMF = 1
  radius = min(int(sigma * 4 * SMF), 3 * len(x))
  channels = x.shape[1]

  kernel = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
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
