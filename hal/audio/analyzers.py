import torch
import scipy.signal
import librosa as rosa
import numpy as np

def onsets(track, fmin=80, fmax=2000):
  onset_env = rosa.onset.onset_strength(track.audio, sr=track.sr, fmin=fmin, fmax=fmax)
  times = rosa.frames_to_time(np.arange(len(onset_env)))
  onset_frames = rosa.onset.onset_detect(onset_envelope=onset_env, hop_length=2048, sr=22050)
  return onset_frames

def chroma(track, n_frames=None, margin=16, notes=12):
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
    chroma = rosa.feature.chroma_cens(y_harm, sr=track.sr)
    if n_frames != None:
      chroma = scipy.signal.resample(chroma, n_frames)
    notes_indices = np.argsort(np.median(chroma, axis=0))[:notes]
    chroma = chroma[:, notes_indices]
    chroma = torch.from_numpy(chroma / chroma.sum(1)[:, None]).float()
    return chroma

def chromagram(track, harmonic_separation_margin=4):
  harmonic = rosa.effects.harmonic(track.audio, margin=harmonic_separation_margin)
  ch = rosa.feature.chroma_cens(harmonic, track.sr)
  return ch
