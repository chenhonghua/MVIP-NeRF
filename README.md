# MVIP-NeRF: Multi-view 3D Inpainting on NeRF Scenes via Diffusion Prior

### Introduction
This repository is for our CVPR 2024 paper '[MVIP-NeRF: Multi-view 3D Inpainting on NeRF Scenes via Diffusion Prior](https://chenhonghua.github.io/clay.github.io/)'. 

## Quick Start

### Dependencies and Environment
```
conda create -n MVIPnerf python=3.8
conda activate MVIPnerf
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
    └── Depth
        ├── IMG_2707.png
        ├── IMG_2708.png
        ├── ...
        └── IMG_2736.png

```
Also, for easier usage of the SPIn-NeRF dataset, we have uploaded one example. Note that our method does not rely on explicit 2D inpaintings results, although we provided the inpainted inputs.
Please click [here](https://drive.google.com/drive/folders/1cSX3nOd73dbLocVH__WjDcUy8Kr9wkyE?usp=sharing) to download the used dataset. Note that if you run our code, just leave the masked area black.
If you use them, please cite SPIn-NeRF and Remove-NeRF.

### Quick Running
```
python DS_NeRF/run.py --config DS_NeRF/config/config_1.txt
```

### Key parameters in the config file
```
datadir: folder for the dataset
factor: downscale of the image resolution of the inpainted scene 
is_normal_guidance: control whether using normal guidance
is_colla_guidance: control whether using multi-view guidance
text: text prompt for the inpainted scene
normalmap_render_factor: we use a factor to downscale the rendered normal map, due to the RAM limitation
```

## TODO
- [x] Release the code.
- [ ] Release video results.


## Acknowledgement
The repository is based on [SPIn-NeRF](https://github.com/SamsungLabs/SPIn-NeRF) and [stable dreamfusion](https://github.com/ashawkey/stable-dreamfusion) 

## License
This project is licensed under NTU S-Lab License 1.0. Redistribution and use should follow [this license](https://github.com/chenhonghua/MVIP-NeRF/blob/main/LICENSE).

# BibTeX
If you find our MVIP-NeRF useful in your work, please consider citing it:
```
@inproceedings{MVIPNeRF,
      title={MVIP-NeRF: Multi-view 3D Inpainting on NeRF Scenes via Diffusion Prior}, 
      author={Honghua Chen and Chen Change Loy and Xingang Pan},
      year={2024},
      booktitle={CVPR},
}
```
