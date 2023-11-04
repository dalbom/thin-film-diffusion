import os
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
from sklearn.utils import shuffle


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

    def _read_from_csv(self, filepath, is_train=True):
        data = pd.read_csv(filepath)

        if is_train:
            columns = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]

            self.data_mean = {c: np.mean(data[c]) for c in columns}
            self.data_std = {c: np.mean(data[c]) for c in columns}

            print("PMMA statistics")
            print(self.data_mean)
            print(self.data_std)
        # else:
        #     data = data.sample(n=128, random_state=np.random.RandomState())
        #     data.to_csv("data/PMMA_sampled_test.csv", index=False)

        filepaths = data["filepath"]

        # Normalize measurements
        measurements = np.stack(
            (
                (data["x1"] - self.data_mean["x1"]) / self.data_std["x1"],
                (data["x2"] - self.data_mean["x2"]) / self.data_std["x2"],
                (data["x3"] - self.data_mean["x3"]) / self.data_std["x3"],
                (data["x4"] - self.data_mean["x4"]) / self.data_std["x4"],
                (data["x5"] - self.data_mean["x5"]) / self.data_std["x5"],
                (data["x6"] - self.data_mean["x6"]) / self.data_std["x6"],
                (data["x7"] - self.data_mean["x7"]) / self.data_std["x7"],
                (data["x8"] - self.data_mean["x8"]) / self.data_std["x8"],
            ),
            axis=1,
        )

        # # Temporal code. Should remove later.
        # if not is_train:
        #     print(filepaths.shape, measurements.shape)
        #     filepaths, measurements = shuffle(filepaths, measurements)
        #     filepaths = filepaths[:128]
        #     measurements = measurements[:128]
        #     print(filepaths.shape, measurements.shape)

        return filepaths, measurements

    def _split_generators(self, dl_manager):
        train_path = "data\\PMMA_train.csv"
        test_path = "data\\PMMA_sampled_test.csv"
        inference_path = "results\\20231104-011749\\inference"

        self.filepaths, self.measurements = self._read_from_csv(train_path)
        self.filepaths_test, self.measurements_test = self._read_from_csv(
            test_path, is_train=False
        )
        self.filepaths_inference = [
            os.path.join(inference_path, f"{i:04d}.png") for i in range(128)
        ]

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
            SplitGenerator(
                name="inference",
                gen_kwargs={
                    "filepaths": self.filepaths_inference,
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
