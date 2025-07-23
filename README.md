# BTP_medMNIST_DP_privacy

## 1.Download miniconda

Go to the official Miniconda download page: https://docs.conda.io/en/latest/miniconda.html

## 2.Project Directory

```
    Your_User_Directory/
    ├── Miniconda3/ <-- Where Miniconda is installed
    │ └── envs/
    │ └── fl_medimage/ <-- Your Conda environment lives here (contains Python.exe,site-packages etc.)
    │ └── (conda environment files)
    │
    └── Projects/ <-- A common place to put all your project folders
    └── fl_medical_image/ <-- This is where your project code goes
    ├── (other .py files for FL client/server, utils)
    ├── (data folder, if you download data there)
    ├── README.md
    └── etc.
```

Example: Install Miniconda in C disk.Go to Desktop and create conda environment.Then place your project folder in the Desktop directory only same directory as the conda environment.

### NOTE: Make sure you are using Miniconda3 command prompt not the Windows PowerShell or Windows Terminal for running below commands.

## 3.Create conda environment

```
conda create -n fl_medimage python=3.10 -y
conda activate fl_medimage
```

## 4.After environment activation

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install medmnist matplotlib scikit-learn tqdm opacus
```

## 5.Installation Verification

```
python -c "import torch; print(torch.**version**)"
python -c "import medmnist; print(medmnist.**version**)"
python -c "import opacus; print(opacus.**version**)"
```

## 6.Run the code

```
cd fl_medical_image     <--- Your project directory
python main.py --privacy dp -- clients 4 --rounds 10    <--- You can choose none also as privacy and can change the clients and no.of training rounds.
```
