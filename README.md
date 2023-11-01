# Thin Film Diffusion Model

[노션 페이지 참고](https://dalbom.notion.site/Thin-Film-Diffusion-Model-52358c026f0345fa8c382065133565fc?pvs=4)

# Data preparation
> The dataset structure is different for each material. Should be formatted in the same structure later.

Lauching the following code will generate one combined csv for the chosen material dataset. 

`python data\combine_data.py --material MATERIAL --path PATH [--output CSV_PATH]`

For example, the combined csv file for the PMMA material will be generated at .\data\PMMA.csv using this code:

`python data\combine_data.py --material PMMA --path c:\dev\ThinFilm\PMMA`

Note that I manually removed '(0.001)' at the end of each directory in the PS dataset because it is a redundant information.

Then you can generate train/test splits for the dataset using this command:

`python data\split_dataset.py --csv_path data\MATERIAL.csv`

# Training

1. Create a virtual environment.

    `conda create --name thinfilm python=3.10`

2. Activate the environment.

    `conda activate thinfilm`

3. Install required packages. Note that the library versions described in requirements.txt are not mandatory. Higher versions may also work.

    `pip install -r requirements.txt`

4. Launch the training script

    `.\run_diffusion_training.bat`