# Neural Style Transfer with VGG19

## Description
This code provides a **PyTorch implementation** of the paper:  
"[Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)",  
by **Leon A. Gatys et al.**  

The application can be run locally either through a **Streamlit application** (`app.py`) or by modifying the input arguments of a **Python script** (`main.py`).

## Brief Overview
**Neural Style Transfer (NST)** is a deep learning technique that generates an image combining:
- the **content** of one image (*content image*),
- with the **style** of another image (*style image*).

## Installation with Conda
These instructions assume that **Anaconda** or **Miniconda** is already installed on your machine.

1. Open **Anaconda Prompt** and clone this repository to your desired location:
   ```bash
   cd <your_folder>
   git clone https://github.com/thiernodaoudaly/neural-style-transfer-app
   cd neural-style-transfer-app
   ```
2. Create the environment using the dependencies provided in env.yml:
   ```bash
   conda env create -f env.yml
   ```
3. Activate the environment:
   ```bash
   conda activate nst-env
   ```