# Thin Film Diffusion Model

[노션 페이지 참고](https://dalbom.notion.site/Thin-Film-Diffusion-Model-52358c026f0345fa8c382065133565fc?pvs=4)

# Data preparation
> The dataset structure is different for each material. Should be formatted in the same structure in the future.

Lauching the following code will generate one combined csv for the chosen material dataset. 

`python data\combine_data.py --material MATERIAL --path PATH [--output CSV_PATH]`

For example, the combined csv file for the PMMA material will be generated at .\data\PMMA.csv using this code:

`python data\combine_data.py --material PMMA --path c:\dev\ThinFilm\PMMA`

Uncomment `rename_files(dataset_path)` is recommended for the first time if you want the filenames to be '%04d.png'.

Note that I manually removed '(0.001)' at the end of each directory in the PS dataset because it is a redundant information.

Then you can generate train/test splits for the dataset using this command:

`python data\split_dataset.py --csv_path data\MATERIAL.csv`

# Training

## Diffusion model

> The following codes and scripts are tested in Windows environment. Modify them accordingly if you use Linux.

1. Create a virtual environment.

    `conda create --name thinfilm python=3.10`

2. Activate the environment.

    `conda activate thinfilm`

3. Install required packages. Note that the library versions described in requirements.txt are not mandatory. Higher versions may also work.

    `pip install -r requirements.txt`

4. Modify the training script if needed. Important parameters are listed here:

    - dataset_name: same as the MATERIAL.
    - resolution: indicates the image size.
    - train_batch_size: batch size for the training process.
    - eval_batch_size: batch size for the validation process.
    - num_epochs: the number of epochs for the training process. 
    - gradient_accumulation_steps: the number of steps for the gradient accumulation
    - learning_rate: the learning rate of the optimizer

    If you encounter CUDA_OUT_OF_MEMORY error, try to adjust **train_batch_size** and **gradient_accumulation_steps** first. Their product should be either 16 or 32. For some GPUS with smaller VRAM (less than 12GB), one can try to lower the **resolution** as well.

5. Launch the training script

    `.\run_diffusers.bat`

6. Launch the tensorboard to monitor the training process

    `tensorboard --logdir results\`

## Regression model

Argument parsing is not implemented yet. Please modify the path to the generated images at a relevant dataset class definition. After then, training a regression model can be initiated using:

`python train_regression.py`

# Inference

## Diffusion model

In order to generate synthetic thin film images from a list of parameters, you can run the following script:

`.\run_inference.bat`

The checkpoint of the diffusion model should be delivered within the script using --ckpt argument.