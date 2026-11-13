# Third-party materials

This project is released under CC BY-NC-SA 4.0 (see [LICENSE](LICENSE)). It builds on the materials below. Each keeps its own license.

## Included in this repository

| Material | Where | License | Owner |
|---|---|---|---|
| MediaPipe Face Landmarker model | `models/face_landmarker.task` | Apache-2.0 | Google |

## Used to train the model

| Material | Use | License | Reference |
|---|---|---|---|
| Flickr-Faces-HQ (FFHQ) | Negative examples: ordinary glasses and bare faces. Only eye-band crops are redistributed, in the dataset release. | CC BY-NC-SA 4.0 for the dataset. Each photo also carries its own license (CC BY 2.0, CC BY-NC 2.0, Public Domain Mark 1.0, CC0 1.0 or U.S. Government Works). | NVIDIA Corporation, https://github.com/NVlabs/ffhq-dataset |
| FFHQ eyeglasses annotations | Deciding which FFHQ photos contain glasses. The annotation files are not redistributed. | CC BY-NC-ND 4.0 | D. Matuzevičius, Zenodo, https://doi.org/10.5281/zenodo.14252074 |
| ResNet18 pretrained weights (`resnet18.a1_in1k`) | Starting point for both branches of the model | Apache-2.0, trained on ImageNet-1k | Ross Wightman, PyTorch Image Models (timm) |
| YouTube videos and image search results | Positive examples and look-alike negative examples. Only eye-band crops are redistributed, in the dataset release. | Copyright of each creator. Used here for non-commercial research. | Lists in `sources/` |

FFHQ attribution for each photo (author, source address and license) is in the official metadata file `ffhq-dataset-v1.json`. The FFHQ image number is kept in the file name of every crop that comes from FFHQ.

Papers to cite when using the FFHQ parts:

- T. Karras, S. Laine, T. Aila. A Style-Based Generator Architecture for Generative Adversarial Networks. CVPR 2019.
- D. Matuzevičius. Diverse Dataset for Eyeglasses Detection: Extending the Flickr-Faces-HQ (FFHQ) Dataset. Sensors 24(23), 7697, 2024. https://doi.org/10.3390/s24237697
- D. Matuzevičius. A Retrospective Analysis of Automated Image Labeling for Eyewear Detection Using Zero-Shot Object Detectors. Electronics 13(23), 4763, 2024. https://doi.org/10.3390/electronics13234763

## Software

| Software | License |
|---|---|
| PyTorch, torchvision | BSD-3-Clause |
| PyTorch Image Models (timm) | Apache-2.0 |
| MediaPipe | Apache-2.0 |
| OpenCV | Apache-2.0 |
| Pillow | MIT-CMU |
| safetensors | Apache-2.0 |
| glasses-detector | MIT |
| yt-dlp | Unlicense |

## Trademarks

Ray-Ban, Meta, Oakley, Rokid, Xiaomi, Samsung, Gentle Monster, Warby Parker, Google, Even Realities and XREAL are trademarks of their owners. They are named here only to say which products the model recognizes. This project is not affiliated with or endorsed by any of them.
