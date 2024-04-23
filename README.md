# MVIP-NeRF: Multi-view 3D Inpainting on NeRF Scenes via Diffusion Prior

### Introduction
This repository is for our CVPR 2024 paper '[MVIP-NeRF: Multi-view 3D Inpainting on NeRF Scenes via Diffusion Prior](https://chenhonghua.github.io/clay.github.io/)'. 

## Quick Start

### Dependencies and Environment
```
conda create -n Pnerf python=3.8
conda activate Pnerf
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install -r requirements_df.txt
pip install lpips
pip install ConfigArgParse
```

### Dataset preparation
Take SPIn-NeRF dataset as example:
```
1
├── images
│   ├── IMG_2707.jpg
│   ├── IMG_2708.jpg
│   ├── ...
│   └── IMG_2736.jpg
└── images_4
    ├── IMG_2707.png
    ├── IMG_2708.png
    ├── ...
    ├── IMG_2736.png
    └── label
        ├── IMG_2707.png
        ├── IMG_2708.png
        ├── ...
        └── IMG_2736.png
    └── Depth_inpainted
        ├── IMG_2707.png
        ├── IMG_2708.png
        ├── ...
        └── IMG_2736.png

```
Also, for easier usage of the SPIn-NeRF dataset, we have uploaded one example. Note that our method does not rely on explicit 2D inpaintings results.
