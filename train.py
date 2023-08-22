import torch
from model.conditioned_diffusion import (
    ConditionedUnet,
    ConditionedDiffusion,
    ConditionedTrainer,
)


def main():
    model = ConditionedUnet(dim=64, dim_mults=(1, 2, 4, 8), channels=1)
    diffusion = ConditionedDiffusion(
        model=model,
        image_size=256,
        timesteps=1000,
        loss_type="l1",
    )
    trainer = ConditionedTrainer(
        diffusion,
        "data\\thin_film.csv",
        train_batch_size=8,
        train_lr=2e-5,
        train_num_steps=700000,
        gradient_accumulate_every=4,
        ema_decay=0.995,
        amp=True,
        calculate_fid=True,
        image_offset=(678, 0, 767, 2452),
        save_and_sample_every=1000,
        num_samples=4,
    )
    trainer.train()


if __name__ == "__main__":
    main()
