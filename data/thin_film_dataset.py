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

        return DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        gt_path = "data\\thin_film.csv"

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

        # self.measurements = np.stack(
        #     (
        #         data["x1"],
        #         data["x2"],
        #         data["x3"],
        #         data["x4"],
        #         data["x5"],
        #         data["x6"],
        #         data["x7"],
        #         data["x8"],
        #     ),
        #     axis=1,
        # )

        return [
            SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"index_range": [0, len(self.filepaths)]},
            )
        ]

    def _generate_examples(self, index_range):
        start, end = index_range

        for i in range(start, end):
            yield i - start, {
                "image": self.filepaths[i],
                "condition": self.measurements[i],
            }
