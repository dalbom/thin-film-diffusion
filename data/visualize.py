import cv2
import os
import numpy as np


def enlarge_results(workdir, fx, fy, samples_per_row, buffer=20):
    filenames = os.listdir(workdir)

    for filename in filenames:
        if "png" not in filename or "enlarged" in filename:
            continue

        img = cv2.imread(os.path.join(workdir, filename))
        img = cv2.resize(img, dsize=None, fx=fx, fy=fy)

        height, width, _ = img.shape
        sample_width = width // samples_per_row

        newimg_height = (height + buffer) * samples_per_row - buffer
        newimg = np.zeros((newimg_height, sample_width, 3), img.dtype)

        for i in range(samples_per_row):
            start_row = i * (height + buffer)
            end_row = start_row + height
            start_col = i * sample_width
            end_col = start_col + sample_width

            newimg[start_row:end_row, :] = img[:, start_col:end_col, :]

        cv2.imwrite(os.path.join(workdir, "enlarged_" + filename), newimg)


def main():
    enlarge_results("results\\diffusers", 2 * 3.196870925684485, 2, 4)
    enlarge_results("results\\diffusers_512", 3.196870925684485, 1, 2)


if __name__ == "__main__":
    main()
