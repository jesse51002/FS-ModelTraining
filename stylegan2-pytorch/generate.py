import os
import argparse

import torch
from torchvision import utils
from tqdm import tqdm


def generate(args, g_ema, device, mean_latent, save_file_name):
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )

            utils.save_image(
                sample,
                os.path.join("sample", save_file_name + ".png"),
                nrow=int(args.sample ** 0.5),
                normalize=True,
                value_range=(-1, 1),
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=36,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=1, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument('--arch', type=str, default='swagan', help='model architectures (stylegan2 | swagan)')

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    if args.arch == 'stylegan2':
        from model import Generator

    elif args.arch == 'swagan':
        from swagan import Generator

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    def run(ckpt, save_folder):
        checkpoint = torch.load(ckpt)
        
        g_ema.load_state_dict(checkpoint["g_ema"])
    
        if args.truncation < 1:
            with torch.no_grad():
                mean_latent = g_ema.mean_latent(args.truncation_mean)
        else:
            mean_latent = None
    
        generate(args, g_ema, device, mean_latent, save_folder)
    
    if args.ckpt is None:
        for ckpt_name in os.listdir("checkpoint/"):
            if not ckpt_name.endswith(".pt"):
                continue
            
            ckpt_pth = "checkpoint/" + ckpt_name
            print(f"Generating for {ckpt_name}")
            run(ckpt_pth, os.path.splitext(ckpt_name)[0])
    else:
        run(args.ckpt, os.path.splitext(os.path.basename(args.ckpt))[0])
        
        
