import os
import pandas as pd
import random
import argparse
from sklearn.model_selection import train_test_split


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Split dataset based on image directories."
    )
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to the CSV file."
    )

    args = parser.parse_args()
    csv_path = args.csv_path

    # Generate output file names based on the input CSV file name
    base_dir, file_name = os.path.split(csv_path)
    base_name, _ = os.path.splitext(file_name)
    train_file_name = os.path.join(base_dir, f"{base_name}_train.csv")
    test_file_name = os.path.join(base_dir, f"{base_name}_test.csv")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Extract unique directories from the 'filepath' column
    directories = df["filepath"].apply(lambda x: x.split("\\\\")[-2]).unique()

    # Shuffle the directories for a fair split
    random.shuffle(directories)

    # Manually handle edge cases with a small number of directories
    if len(directories) < 5:
        train_dirs = directories[:-1]  # All except the last one
        test_dirs = directories[-1:]  # Only the last one
    else:
        # Split directories into training and test sets
        train_dirs, test_dirs = train_test_split(
            directories, test_size=0.2, random_state=42
        )  # Adjust test_size as needed

    # Filter rows based on the directories for training and test sets
    train_df = df[df["filepath"].apply(lambda x: x.split("\\\\")[-2] in train_dirs)]
    test_df = df[df["filepath"].apply(lambda x: x.split("\\\\")[-2] in test_dirs)]

    # Save the training and test datasets into separate CSV files
    train_df.to_csv(train_file_name, index=False)
    test_df.to_csv(test_file_name, index=False)


if __name__ == "__main__":
    main()
