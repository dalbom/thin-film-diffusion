import math
import datetime
from functools import partial
from pathlib import Path
from random import random

from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from ema_pytorch import EMA
from PIL import Image
from pytorch_fid.inception import InceptionV3
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import utils
from tqdm import tqdm

from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    ModelPrediction,
    cpu_count,
    cycle,
    default,
    exists,
    extract,
    has_int_squareroot,
    identity,
    num_to_groups,
    reduce,
)


class ConditionedUnet(Unet):
    def __init__(self, dim, **kwargs):
        super().__init__(dim=dim, **kwargs)

        # The size of the time embedding is defined as dim * 4
        self.embed_condition = torch.nn.Linear(8, dim * 4)
        self.combine_ct = torch.nn.Sequential(
            torch.nn.GELU(), torch.nn.Linear(dim * 8, dim * 4)
        )

    def forward(self, x, condition, time):
        x = self.init_conv(x)
        r = x.clone()

        c = self.embed_condition(condition)
        t = self.time_mlp(time)

        t = self.combine_ct(torch.cat((c, t), dim=1))

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)

        return self.final_conv(x)


class ConditionedDiffusion(GaussianDiffusion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def model_predictions(
        self, x, condition, t, clip_x_start=False, rederive_pred_noise=False
    ):
        model_output = self.model(x, condition, t)
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, condition, t, clip_denoised=True):
        preds = self.model_predictions(x, condition, t)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, condition, t: int, x_self_cond=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, condition=condition, t=batched_times, clip_denoised=True
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, condition, shape, return_all_timesteps=False):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img, _ = self.p_sample(img, condition, t)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, condition, batch_size=16, return_all_timesteps=False):
        image_size, channels = self.image_size, self.channels
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        )
        return sample_fn(
            condition,
            (batch_size, channels, image_size, image_size),
            return_all_timesteps=return_all_timesteps,
        )

    @torch.no_grad()
    def ddim_sample(self, condition, shape, return_all_timesteps=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img, condition, time_cond, clip_x_start=True, rederive_pred_noise=True
            )

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    def p_losses(self, x_start, condition, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # # and condition with unet with that
        # # this technique will slow down training by 25%, but seems to lower FID significantly

        # x_self_cond = None
        # if self.self_condition and random() < 0.5:
        #     with torch.no_grad():
        #         x_self_cond = self.model_predictions(x, condition, t).pred_x_start
        #         x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, condition, t)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = self.loss_fn(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, condition, *args, **kwargs):
        (
            b,
            c,
            h,
            w,
            device,
            img_size,
        ) = (
            *img.shape,
            img.device,
            self.image_size,
        )
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, condition, t, *args, **kwargs)


class Crop(torch.nn.Module):
    def __init__(self, top, left, height, width):
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, img):
        return TF.crop(img, self.top, self.left, self.height, self.width)


class ConditionedDataset(Dataset):
    def __init__(self, gt_path, image_offset, image_size):
        super().__init__()

        top, left, height, width = image_offset
        self.paths = self._load_data(gt_path)

        self.transform = T.Compose(
            [
                Crop(top, left, height, width),
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]
        )

    def _load_data(self, gt_path):
        data = pd.read_csv(gt_path)

        self.filepaths = data["filepath"]
        self.measurements = np.stack(
            (
                (data["x1"] - np.mean(data["x1"])) / np.std(data["x1"]),
                (data["x2"] - np.mean(data["x2"])) / np.std(data["x2"]),
                (data["x3"] - np.mean(data["x3"])) / np.std(data["x3"]),
                (data["x4"] - np.mean(data["x4"])) / np.std(data["x4"]),
                (data["x5"] - np.mean(data["x5"])) / np.std(data["x5"]),
                (data["x6"] - np.mean(data["x6"])) / np.std(data["x6"]),
                (data["x7"] - np.mean(data["x7"])) / np.std(data["x7"]),
                (data["x8"] - np.mean(data["x8"])) / np.std(data["x8"]),
            ),
            axis=1,
        )

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        img = Image.open(self.filepaths[index])

        return self.transform(img), torch.Tensor(self.measurements[index])


class ConditionedTrainer(Trainer):
    def __init__(
        self,
        diffusion_model,
        gt_path,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder="./results",
        amp=False,
        fp16=False,
        split_batches=True,
        image_offset=None,
        calculate_fid=True,
        inception_block_idx=2048,
    ):
        # log writer
        # Get the current date and time
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Create the SummaryWriter with the desired folder name
        self.log_writer = SummaryWriter(f"results\\{current_time}")

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches, mixed_precision="fp16" if fp16 else "no"
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # InceptionV3 for fid-score computation

        self.inception_v3 = None

        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)

        # sampling and training hyperparameters

        assert has_int_squareroot(
            num_samples
        ), "number of samples must have an integer square root"
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        self.ds = ConditionedDataset(
            gt_path,
            image_offset,
            self.image_size,
        )
        dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=cpu_count() - 1,
        )

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model, beta=ema_decay, update_every=ema_update_every
            )
            self.ema.to(self.device)

        self.results_folder = Path(f"results\\{current_time}")
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    batch_img, batch_condition = next(self.dl)

                    batch_img = batch_img.to(device)
                    batch_condition = batch_condition.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(batch_img, batch_condition)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                self.log_writer.add_scalar("loss/train", total_loss, self.step)
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(
                                map(
                                    lambda n: self.ema.ema_model.sample(
                                        batch_condition[: self.num_samples],
                                        batch_size=n,
                                    ),
                                    batches,
                                )
                            )

                        all_images = torch.cat(all_images_list, dim=0)

                        utils.save_image(
                            all_images,
                            str(self.results_folder / f"sample-{milestone}.png"),
                            nrow=int(math.sqrt(self.num_samples)),
                        )
                        self.save(milestone)

                        # whether to calculate fid

                        if exists(self.inception_v3):
                            fid_score = self.fid_score(
                                real_samples=batch_img, fake_samples=all_images
                            )
                            accelerator.print(f"fid_score: {fid_score}")

                pbar.update(1)

        accelerator.print("training complete")
