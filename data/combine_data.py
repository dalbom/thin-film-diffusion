import os
import pandas as pd


def rename_files(path):
    filenames = os.listdir(path)

    for filename in filenames:
        if "tif" not in filename:
            continue

        idx = int(filename.split(".")[0])

        src = os.path.join(path, filename)
        dst = os.path.join(path, "{:04d}.tif".format(idx))

        os.rename(src, dst)


writer = open("data\\thin_film.csv", "w")
writer.write("filepath,x1,x2,x3,x4,x5,x6,x7,x8\n")
WORKDIR = "C:\\dev\\dataset\\ThinFilm"

datasets = sorted(os.listdir(WORKDIR))

for dataset in datasets:
    dataset_path = os.path.join(WORKDIR, dataset)

    if not os.path.isdir(dataset_path) or "data" not in dataset:
        continue

    # rename_files(dataset_path)

    gt_filepath = os.path.join(dataset_path, dataset + ".xlsx")
    dataframe = pd.read_excel(gt_filepath, engine="openpyxl")

    for index, row in dataframe.iterrows():
        image_id = int(row["image#"])
        values = [
            row["x1"],
            row["x2"],
            row["x3"],
            row["x4"],
            row["x5"],
            row["x6"],
            row["x7"],
            row["x8"],
        ]
        image_path = os.path.join(dataset_path, "{:04d}.tif".format(image_id))
        image_path = [image_path.replace("\\", "\\\\")]
        values = image_path + [str(v) for v in values]

        writer.write(",".join(values) + "\n")

writer.flush()
writer.close()
