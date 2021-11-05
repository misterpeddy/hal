from abc import ABC, abstractmethod

from hal import gen
from hal import io_utils

class BaseScene(ABC):

  def __init__(self, model, track):
    self.model = model
    self.track = track
    self.n_frames = gen.VID_FRAME_RATE * int(track.audio.shape[0] / track.sr)
    self.gen_dir = io_utils.create_cache_dir('gen', 'video')

  #@hal.remote
  @abstractmethod
  def render(self):
    return gen.render(self)

class ChromaScene(BaseScene):
  """Generates chroma-weighted latents with onset-weighted noise"""
  def render(self):
    self.latents = gen.chroma_latents(self.model, self.track)
    self.noise = gen.noise(self.model, self.track)
    return super().render()

class InterpolationScene(BaseScene):
  """Interpolates between two latents"""
  def render(self):
    self.latents = gen.get_interp_onsets_latents(self.model, self.track)
    self.noise = gen.noise(self.model, self.track)
    return super().render()

class MorphingInterpolationScene(BaseScene):
  """Interpolates between two latents, with onset weighted movement"""
  def render(self):
    self.latents = gen.get_interp_onsets_latents_with_movement(self.model, self.track)
    self.noise = gen.empty_noise(self.model)
    return super().render()
