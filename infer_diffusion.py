import argparse
import inspect
import os

import numpy as np
import torch
from datasets import load_dataset
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset_pmma import Crop
from model.pipeline import ConditionalPipeline
from model.unet import ConditionalUNet


def load_model_from_checkpoint_old(checkpoint_path, device):
    # Load the model from the checkpoint
    model = (
        ConditionalUNet()
    )  # This needs to match the actual model class used in training
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def load_model_from_checkpoint_old2(checkpoint_path, device):
    default_resolution = 256

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize the model
    model = ConditionalUNet(
        sample_size=default_resolution,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )  # This needs to match the actual model class used in training

    # Assuming the EMA weights are stored in the checkpoint under 'ema_state_dict'
    ema_state_dict = checkpoint.get(
        "ema_state_dict", checkpoint["model_state_dict"]
    )  # Fallback to regular state dict if EMA is not present

    # Load the EMA weights
    model.load_state_dict(ema_state_dict)
    model.eval()
    return model


def load_model_from_checkpoint(input_dir, use_ema, device):
    default_resolution = 256

    model = ConditionalUNet(
        sample_size=default_resolution,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    if use_ema:
        # Load the EMA model from the pretrained directory
        ema_model = EMAModel.from_pretrained(
            os.path.join(input_dir, "unet_ema"), ConditionalUNet
        )
        ema_model.copy_to(model.parameters())
    else:
        # Load the regular model from the pretrained directory
        model = ConditionalUNet.from_pretrained(input_dir, subfolder="unet")

    model.to(device)
    model.eval()
    return model


def main(input_dir, output_path):
    # list of default parameters
    default_ddpm_num_steps = 1000
    default_ddpm_beta_schedule = "linear"
    default_prediction_type = "epsilon"
    default_resolution = 256
    default_use_ema = True
    default_batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model_from_checkpoint(input_dir, default_use_ema, device)

    # Define the pipeline with the UNet model and the scheduler
    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(
        inspect.signature(DDPMScheduler.__init__).parameters.keys()
    )
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=default_ddpm_num_steps,
            beta_schedule=default_ddpm_beta_schedule,
            prediction_type=default_prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=default_ddpm_num_steps,
            beta_schedule=default_ddpm_beta_schedule,
        )

    pipeline = ConditionalPipeline(unet=model, scheduler=noise_scheduler)

    # Load the dataset
    test_dataset = load_dataset(
        "data\\dataset_pmma.py",
        split="test",
    )

    crop_info = [int(v) for v in test_dataset.description.split(",")]

    augmentations_test = transforms.Compose(
        [
            Crop(*crop_info),
            transforms.Resize(
                (default_resolution, default_resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )

    def transform_images_test(examples):
        images = [augmentations_test(Image.open(image)) for image in examples["image"]]

        return {"image": images, "condition": examples["condition"]}

    def collate_fn(examples):
        images = torch.stack([example["image"] for example in examples])

        conditions = torch.stack(
            [torch.Tensor(example["condition"]) for example in examples]
        )

        return {"image": images, "condition": conditions}

    test_dataset.set_transform(transform_images_test)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=default_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )  # Batch size is 1 as requested

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Generate and save images
    for i, batch in enumerate(test_dataloader):
        conditions = batch["condition"].to(device)

        # You might want to change the seed for each batch
        generator = torch.Generator(device=device).manual_seed(0)
        images_np = pipeline(
            condition=conditions,
            generator=generator,
            batch_size=default_batch_size,
            num_inference_steps=default_ddpm_num_steps,
            output_type="numpy",
        ).images

        images_np = (images_np * 255).round().astype("uint8")

        # Save each image in the batch separately using PIL
        for j, image_np in enumerate(images_np):
            image_idx = (
                i * default_batch_size + j
            )  # Calculate the index of the image in the entire dataset
            image_pil = Image.fromarray(np.squeeze(image_np))
            image_pil.save(os.path.join(output_path, f"{image_idx:04d}.png"))

        # # Save each image in the batch separately
        # for j, image in enumerate(images_tensor):
        #     # Calculate the index of the image in the entire dataset
        #     image_idx = i * default_batch_size + j
        #     # Save the image
        #     save_image(image, os.path.join(output_path, f"{image_idx:04d}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to the model checkpoint"
    )

    args = parser.parse_args()

    output_path = os.path.join(args.ckpt, "..", "inference")

    main(args.ckpt, output_path)
