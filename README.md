# MVIP-NeRF: Multi-view 3D Inpainting on NeRF Scenes via Diffusion Prior

### Introduction
This repository is for our CVPR 2024 paper '[MVIP-NeRF: Multi-view 3D Inpainting on NeRF Scenes via Diffusion Prior](https://chenhonghua.github.io/clay.github.io/)'. 

## Quick Start

### Dependencies
```
conda create -n Pnerf python=3.8
conda activate Pnerf
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install -r requirements_df.txt
pip install lpips
pip install ConfigArgParse
```
