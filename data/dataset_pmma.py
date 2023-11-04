import datasets
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from datasets import (
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Sequence,
    SplitGenerator,
    Value,
)


class Crop(torch.nn.Module):
    def __init__(self, top, left, height, width):
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, img):
        return TF.crop(img, self.top, self.left, self.height, self.width)


class ThinFilmDataset(GeneratorBasedBuilder):
    VERSION = "0.1.0"

    def _info(self):
        features = Features(
            {"image": Value("string"), "condition": Sequence(Value("float"))}
        )

        return DatasetInfo(features=features, description="678,0,767,2452")

    def _read_from_csv(self, filepath):
        print(filepath)
        data = pd.read_csv(filepath)

        filepaths = data["filepath"]

        # Normalize measurements
        measurements = np.stack(
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

        return filepaths, measurements

    def _split_generators(self, dl_manager):
        train_path = "data\\PMMA_train.csv"
        test_path = "data\\PMMA_test.csv"

        self.filepaths, self.measurements = self._read_from_csv(train_path)
        self.filepaths_test, self.measurements_test = self._read_from_csv(test_path)

        return [
            SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": self.filepaths,
                    "measurements": self.measurements,
                },
            ),
            SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": self.filepaths_test,
                    "measurements": self.measurements_test,
                },
            ),
        ]

    def _generate_examples(self, filepaths, measurements):
        for i in range(len(filepaths)):
            yield i, {
                "image": filepaths[i],
                "condition": measurements[i],
            }
