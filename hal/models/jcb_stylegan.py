import os
import torch
import sys
import ray
import numpy as np

from torchvision import utils as tutils
from tqdm import tqdm

import hal.io_utils as io_utils
import hal as hal

from third_party.jcbrouwer_maua_stylegan2.models.stylegan2 import Generator as SG2Generator

import matplotlib.pyplot as plt

## for _train
import argparse


#@hal.actor
#@ray.remote
class JCBStyleGan:

  def __init__(self):

    _ckpt = "/content/freagan.pt"
    self.n_mlp = 8
    self.latent_dim = 512
    self.output_size = 1024
    self.generator_resolution = 1024
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.g_ema = SG2Generator(self.output_size, self.latent_dim, self.n_mlp, constant_input=True).to(self.device)
    self.checkpoint = torch.load(_ckpt)
    self.g_ema.load_state_dict(self.checkpoint["g_ema"], strict=False)
    self.gen_dir = "/content/gen/"
    os.makedirs(self.gen_dir, exist_ok=True)

  def _generate_latents(self, n_latents):
    """Generates random, mapped latents

    Args:
        n_latents (int): Number of mapped latents to generate

    Returns:
        torch.tensor: Set of mapped latents
    """
    zs = torch.randn((n_latents, self.latent_dim), device=self.device)
    latent_selection = self.g_ema(zs, map_latents=True)
    return latent_selection
  
  def _code_to_image(self, z=None, w=None):
    assert z != None or w != None
    if z != None:
      out, _ = self.g_ema([z])
    if w != None:
      out, _ = self.g_ema(w, input_is_latent=True)
    plt.figure(figsize=(10 * out.shape[0], 8 * out.shape[0]))
    out = (out.clamp_(-1, 1) + 1) * 127.5
    out = out.permute(0, 2, 3, 1)
    out = out.cpu().numpy().astype(np.uint8)
    out = out.reshape(out.shape[0] * out.shape[1], *out.shape[2:4])
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    plt.imshow(out)

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
    _train(self)

  def project(self, image):
    pass

  def example(self, n=1, n_samples=1, truncation=0.9, truncation_mean=4096, seeds=[], center=None):
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


  def _train(self):
    device = "cuda"

    parser = argparse.ArgumentParser()

    parser.add_argument("--wbname", type=str, required=True)
    parser.add_argument("--wbproj", type=str, required=True)
    parser.add_argument("--wbgroup", type=str, default=None)

    # data options
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--vflip", type=bool, default=False)
    parser.add_argument("--hflip", type=bool, default=True)

    # training options
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_accumulate", type=int, default=1)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--transfer_mapping_only", type=bool, default=False)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--iter", type=int, default=20_000)

    # model options
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--min_rgb_size", type=int, default=4)
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--n_sample", type=int, default=60)
    parser.add_argument("--constant_input", type=bool, default=False)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--d_skip", type=bool, default=True)

    # optimizer options
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--d_lr_ratio", type=float, default=1.0)
    parser.add_argument("--lookahead", type=bool, default=True)
    parser.add_argument("--la_steps", type=float, default=500)
    parser.add_argument("--la_alpha", type=float, default=0.5)

    # loss options
    parser.add_argument("--r1", type=float, default=1e-5)
    parser.add_argument("--path_regularize", type=float, default=1)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing_prob", type=float, default=0.666)

    # augmentation options
    parser.add_argument("--augment", type=bool, default=True)
    parser.add_argument("--contrastive", type=float, default=0)
    parser.add_argument("--balanced_consistency", type=float, default=0)
    parser.add_argument("--augment_p", type=float, default=0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=15_000)

    # validation options
    parser.add_argument("--val_batch_size", type=int, default=6)
    parser.add_argument("--fid_n_sample", type=int, default=2500)
    parser.add_argument("--fid_truncation", type=float, default=None)
    parser.add_argument("--ppl_space", choices=["z", "w"], default="w")
    parser.add_argument("--ppl_n_sample", type=int, default=1250)
    parser.add_argument("--ppl_crop", type=bool, default=False)

    # logging options
    parser.add_argument("--log_spec_norm", type=bool, default=False)
    parser.add_argument("--img_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=-1)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--profile_mem", action="store_true")

    # (multi-)GPU options
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--cudnn_benchmark", type=bool, default=True)

    args = parser.parse_args()
    if args.balanced_consistency > 0 or args.contrastive > 0:
        args.augment = True
    args.name = os.path.splitext(os.path.basename(args.path))[0]
    args.r1 = args.r1 * args.size ** 2

    args.num_gpus = 1
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    args.distributed = False

    # code for updating wandb configs that were incorrect
    # if args.local_rank == 0:
    #     api = wandb.Api()
    #     run = api.run("wav/temperatuur/7kp6g0zt")
    #     run.config = vars(args)
    #     run.update()
    # exit()

    generator = Generator(
        args.size,
        args.latent_size,
        args.n_mlp,
        channel_multiplier=args.channel_multiplier,
        constant_input=args.constant_input,
        min_rgb_size=args.min_rgb_size,
    ).to(device, non_blocking=True)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier, use_skip=args.d_skip).to(
        device
    )

    if args.log_spec_norm:
        for name, parameter in generator.named_parameters():
            if "weight" in name and parameter.squeeze().dim() > 1:
                mod = generator
                for attr in name.replace(".weight", "").split("."):
                    mod = getattr(mod, attr)
                validation.track_spectral_norm(mod)
        for name, parameter in discriminator.named_parameters():
            if "weight" in name and parameter.squeeze().dim() > 1:
                mod = discriminator
                for attr in name.replace(".weight", "").split("."):
                    mod = getattr(mod, attr)
                validation.track_spectral_norm(mod)

    g_ema = Generator(
        args.size,
        args.latent_size,
        args.n_mlp,
        channel_multiplier=args.channel_multiplier,
        constant_input=args.constant_input,
        min_rgb_size=args.min_rgb_size,
    ).to(device, non_blocking=True)
    g_ema.requires_grad_(False)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    if args.contrastive > 0:
        contrast_learner = ContrastiveLearner(
            discriminator,
            args.size,
            augment_fn=nn.Sequential(
                nn.ReflectionPad2d(int((math.sqrt(2) - 1) * args.size / 4)),  # zoom out
                augs.RandomHorizontalFlip(),
                RandomApply(augs.RandomAffine(degrees=0, translate=(0.25, 0.25), shear=(15, 15)), p=0.1),
                RandomApply(augs.RandomRotation(180), p=0.1),
                augs.RandomResizedCrop(size=(args.size, args.size), scale=(1, 1), ratio=(1, 1)),
                RandomApply(augs.RandomResizedCrop(size=(args.size, args.size), scale=(0.5, 0.9)), p=0.1),  # zoom in
                RandomApply(augs.RandomErasing(), p=0.1),
            ),
            hidden_layer=(-1, 0),
        )
    else:
        contrast_learner = None

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = torch.optim.Adam(
        generator.parameters(), lr=args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio * args.d_lr_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.lookahead:
        g_optim = LookaheadMinimax(
            g_optim, d_optim, la_steps=args.la_steps, la_alpha=args.la_alpha, accumulate=args.num_accumulate
        )

    if args.checkpoint is not None:
        print("load model:", args.checkpoint)

        checkpoint = th.load(args.checkpoint, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.checkpoint)
            args.start_iter = int(os.path.splitext(ckpt_name)[-1].replace(args.name, ""))
        except ValueError:
            pass

        if args.transfer_mapping_only:
            print("Using generator latent mapping network from checkpoint")
            mapping_state_dict = {}
            for key, val in checkpoint["state_dict"].items():
                if "generator.style" in key:
                    mapping_state_dict[key.replace("generator.", "")] = val
            generator.load_state_dict(mapping_state_dict, strict=False)
        else:
            generator.load_state_dict(checkpoint["g"], strict=False)
            g_ema.load_state_dict(checkpoint["g_ema"], strict=False)

            discriminator.load_state_dict(checkpoint["d"], strict=False)

            if args.lookahead:
                g_optim.load_state_dict(checkpoint["g_optim"], checkpoint["d_optim"])
            else:
                g_optim.load_state_dict(checkpoint["g_optim"])
                d_optim.load_state_dict(checkpoint["d_optim"])

        del checkpoint
        th.cuda.empty_cache()

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        if contrast_learner is not None:
            contrast_learner = nn.parallel.DistributedDataParallel(
                contrast_learner,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

    transform = transforms.Compose(
        [
            transforms.RandomVerticalFlip(p=0.5 if args.vflip else 0),
            transforms.RandomHorizontalFlip(p=0.5 if args.hflip else 0),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    if get_rank() == 0:
        validation.get_dataset_inception_features(loader, args.name, args.size)
        if args.wbgroup is None:
            wandb.init(project=args.wbproj, name=args.wbname, config=vars(args))
        else:
            wandb.init(project=args.wbproj, group=args.wbgroup, name=args.wbname, config=vars(args))

    if args.profile_mem:
        os.environ["GPU_DEBUG"] = str(args.local_rank)
        from gpu_profile import gpu_profile

        sys.settrace(gpu_profile)

    train(args, loader, generator, discriminator, contrast_learner, g_optim, d_optim, g_ema)

