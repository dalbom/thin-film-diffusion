import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from transformers.optimization import get_scheduler
from accelerate import Accelerator
import os
from torchvision import transforms
from data.thin_film_dataset import Crop
from PIL import Image
from accelerate.utils import ProjectConfiguration
import datetime


class EarlyStopping:
    def __init__(self, patience=3, threshold=0.01):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float("inf")
        self.counter = 0

    def check(self, val_loss):
        if self.best_loss - val_loss > self.threshold:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def is_new_record(self):
        return self.counter == 0


def get_dataloaders():
    # Define dataset, transformations, and batch_size
    # NOTE: You should have train_dataset defined from earlier
    batch_size = 32
    thin_film_dataset = load_dataset(
        "data\\thin_film_dataset.py",
        split="train",
    )

    # Splitting the dataset
    total_size = len(thin_film_dataset)
    train_size = int(0.8 * total_size)
    valid_size = int(0.1 * total_size)
    test_size = total_size - train_size - valid_size
    train_subset, valid_subset, test_subset = random_split(
        thin_film_dataset, [train_size, valid_size, test_size]
    )

    # Apply transform
    resolution = 256
    top, left, height, width = 678, 0, 767, 2452

    # Preprocessing the datasets and DataLoaders creation.
    augmentations = transforms.Compose(
        [
            Crop(top, left, height, width),
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(Image.open(image)) for image in examples["image"]]

        return {"image": images, "condition": examples["condition"]}

    def collate_fn(examples):
        images = torch.stack([example["image"] for example in examples])

        conditions = torch.stack(
            [torch.Tensor(example["condition"]) for example in examples]
        )

        return {"image": images, "condition": conditions}

    thin_film_dataset.set_transform(transform_images)

    train_dataloader = DataLoader(
        train_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )
    test_dataloader = DataLoader(
        test_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )

    return train_dataloader, valid_dataloader, test_dataloader


# Hyperparameters
num_epochs = 100


# Data loaders
train_dataloader, valid_dataloader, test_dataloader = get_dataloaders()

# Use torchvision's resnet18
model = resnet18(weights=None)  # Turn off pretrained
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 8)  # Customize for regression output

# Loss function, optimizer and accelerator
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
project_dir = f"results\\regression\\{current_time}"
logging_dir = os.path.join(project_dir, "logs")
accelerator_project_config = ProjectConfiguration(
    project_dir=project_dir, logging_dir=logging_dir
)
accelerator = Accelerator(
    log_with="tensorboard", project_config=accelerator_project_config
)

# Initialize the learning rate scheduler
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataloader) * num_epochs,
)


# Prepare everything with our accelerator
(
    model,
    optimizer,
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    criterion,
    lr_scheduler,
) = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    criterion,
    lr_scheduler,
)

# Training loop with early stopping
run = os.path.split(__file__)[-1].split(".")[0]
accelerator.init_trackers(run)
tracker = accelerator.get_tracker("tensorboard", unwrap=True)

global_step = 0
early_stopping = EarlyStopping(patience=10, threshold=1e-4)
output_dir = "results\\regression"

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, batch in tqdm(enumerate(train_dataloader)):
        model_input = batch["image"]
        gt = batch["condition"]
        optimizer.zero_grad()

        model_output = model(model_input)
        loss = criterion(model_output, gt)
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        logs = {
            "loss/train": loss.item(),
            "lr": lr_scheduler.get_last_lr()[0],
        }
        accelerator.log(logs, step=global_step)

        global_step += 1

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in valid_dataloader:
            model_input = batch["image"]
            gt = batch["condition"]

            model_output = model(model_input)
            loss = criterion(model_output, gt)
            val_loss += loss.item()

    val_loss /= len(valid_dataloader)
    accelerator.log({"loss/validation": val_loss}, step=global_step)

    if early_stopping.check(val_loss):
        break

    # New minimum validation loss
    if early_stopping.is_new_record():
        test_loss = 0.0

        with torch.no_grad():
            for data in test_dataloader:
                model_input = batch["image"]
                gt = batch["condition"]

                model_output = model(model_input)
                loss = criterion(model_output, gt)
                test_loss += loss.item()

        test_loss /= len(test_dataloader)
        accelerator.log({"loss/test": test_loss}, step=global_step)

        torch.save(
            model.state_dict(), os.path.join(project_dir, f"epoch_{epoch+1:03d}.pth")
        )

print("Finished Training")
