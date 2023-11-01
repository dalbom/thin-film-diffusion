import os
import argparse
import pandas as pd


def rename_files_PMMA(path):
    filenames = os.listdir(path)

    for filename in filenames:
        if "tif" not in filename:
            continue

        idx = int(filename.split(".")[0])

        src = os.path.join(path, filename)
        dst = os.path.join(path, "{:04d}.tif".format(idx))

        os.rename(src, dst)


def rename_files(path):
    filenames = sorted(os.listdir(path))

    for idx, filename in enumerate(filenames):
        if "tif" not in filename:
            continue

        src = os.path.join(path, filename)
        dst = os.path.join(path, "{:04d}.tif".format(idx))

        os.rename(src, dst)


def process_PMMA(WORKDIR, output_path):
    writer = open(output_path, "w")
    writer.write("filepath,x1,x2,x3,x4,x5,x6,x7,x8\n")

    datasets = sorted(os.listdir(WORKDIR))

    for dataset in datasets:
        dataset_path = os.path.join(WORKDIR, dataset)

        if not os.path.isdir(dataset_path) or "data" not in dataset:
            continue

        # Uncomment to rename files if necessary
        # rename_files_PMMA(dataset_path)

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


def process_PS(WORKDIR, output_path):
    columns = [
        "Time (s)",
        "Load (N)",
        "Stress (MPa)",
        "Strain A (-)",
        "Strain B (-)",
        "Distance A (pixel)",
        "Distance B (pixel)",
        "Poisson's ratio (-)",
        "Displacement (mm)",
        "Displacement Velocity (mm/s)",
    ]
    writer = open(output_path, "w")
    writer.write(
        "filepath,time,load,stress,strain_A,strain_B,distance_A,distance_B,Poisson_ratio,displacement,displacement_velocity\n"
    )

    datasets = sorted(os.listdir(WORKDIR))

    for dataset in datasets:
        dataset_path = os.path.join(WORKDIR, dataset)

        if not os.path.isdir(dataset_path):
            continue

        # Uncomment to rename files if necessary
        # rename_files(dataset_path)

        gt_filepath = os.path.join(WORKDIR, dataset + ".xlsx")
        dataframe = pd.read_excel(gt_filepath, engine="openpyxl")

        for index, row in dataframe.iterrows():
            values = [row[c] for c in columns]
            image_path = os.path.join(dataset_path, "{:04d}.tif".format(index))
            image_path = [image_path.replace("\\", "\\\\")]
            values = image_path + [str(v) for v in values]

            writer.write(",".join(values) + "\n")

    writer.flush()
    writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Process datasets of different materials."
    )
    parser.add_argument(
        "--material", type=str, required=True, help="Type of the dataset material."
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Absolute path of the dataset."
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save the processed dataset."
    )

    args = parser.parse_args()

    material = args.material
    WORKDIR = args.path
    output_path = (
        args.output if args.output else os.path.join("./data", f"{material}.csv")
    )

    if material == "PMMA":
        process_PMMA(WORKDIR, output_path)
    elif material == "PS":
        process_PS(WORKDIR, output_path)
    else:
        print(f"Processing for material {material} is not implemented yet.")


if __name__ == "__main__":
    main()
