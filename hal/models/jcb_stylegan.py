import os
import torch
import sys
import ray

from torchvision import utils as tutils
from tqdm import tqdm

import hal_project.hal.io_utils as io_utils
import hal_project.hal as hal

from hal_project.third_party.jcbrouwer_maua_stylegan2.models.stylegan2 import Generator as SG2Generator

@hal.actor
@ray.remote
class JCBStyleGan:

  def __init__(self):

    # sys.path.append('/home/peddy_google_com/hal/old')
    _ckpt = "/home/peddy_google_com/data/abstract_art_000280.pt"
    self.n_mlp = 8
    self.latent_dim = 512
    self.output_size = 512
    self.generator_resolution = 512
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.g_ema = SG2Generator(self.output_size, self.latent_dim, self.n_mlp, constant_input=True).to(self.device)
    self.checkpoint = torch.load(_ckpt)
    self.g_ema.load_state_dict(self.checkpoint["g_ema"], strict=False)
    self.gen_dir = "/home/peddy_google_com/drive/ai_art/stylegan-ada-experiments/00006-train_preprocessed-auto1-resumecustom/gen/"
    os.makedirs(self.gen_dir, exist_ok=True)

  def generate_latents(self, n_latents):
    """Generates random, mapped latents

    Args:
        n_latents (int): Number of mapped latents to generate

    Returns:
        th.tensor: Set of mapped latents
    """
    zs = th.randn((n_latents, self.latent_dim), device=self.device)
    latent_selection = self.g_ema(zs, map_latents=True).cpu()
    return latent_selection

  def get_noise_range(self):
    """Gets the correct number of noise resolutions."""
    log_max_res = int(np.log2(self.output_size))
    log_min_res = 2 + (log_max_res - int(np.log2(self.generator_resolution)))
    range_min = 2 * log_min_res + 1
    range_max = 2 * (log_max_res + 1)
    side_fn = lambda x: int(x / 2)
    return range_min, range_max, side_fn

  def generate_audiovisual(self):
    pass

  def train(self, dataset, finetune=True):
    pass

  def project(self, image):
    pass

  def example(self, n=1, n_samples=1, truncation=0.5, truncation_mean=4096, seeds=[], center=None):
    """ Creates `n_samples` of `n` examples"""
    mean_latent = None
    if truncation < 1:
      with torch.no_grad():
        mean_latent = self.g_ema.mean_latent(truncation_mean)

    gen_dir = io_utils.next_dir(self.gen_dir)
    with torch.no_grad():
      self.g_ema.eval()
      sample_paths = []
      for i in tqdm(range(n)):
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        sample, _ = self.g_ema([z], truncation=truncation, truncation_latent=mean_latent)
        path = os.path.join(gen_dir, f"{str(i).zfill(6)}.png")
        tutils.save_image(sample, path, nrow=1, normalize=True, range=(-1, 1))
        sample_paths.append(path)

    return sample_paths

  def explore(self):
    pass
