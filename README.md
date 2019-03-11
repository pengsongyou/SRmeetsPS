# SRmeetsPS

This repository contains the code for our papers:  
> Songyou Peng, Bjoern Haefner, Yvain Queau and Daniel Cremers, "**[Depth Super-Resolution Meets Uncalibrated Photometric Stereo](https://arxiv.org/abs/1708.00411)**", In IEEE Conference on Computer Vision (ICCV) Workshop, 2017.

and 

> Bjoern Haefner*, Songyou Peng*, Alok Verma*, Yvain Queau and Daniel Cremers, "**[Photometric Depth Super-Resolution](https://arxiv.org/abs/1809.10097)**", arXiv, 2018. (* equally contributed)

A CUDA version code is also available [here](https://github.com/nihalsid/SRmeetsPS-CUDA).

## Input
* Super-resolution RGB images (at least **4** images)
* Super-resolution binary mask
* Low-resolution depth images (1 image is fine, same size as RGB image is also fine)
* Intrinsic matrix (containing the focal length and principle points of the RGB images)
* Downsampling matrix (you can aquire with ``getDownsampleMat.m``. The file name should be like ``D_1280_960_2.mat``)

**All the real-world data can be found at [this link](https://vision.in.tum.de/data/datasets/photometricdepthsr).**

## Requirement
* MATLAB (tested and working in R2015b and later versions)
* [Optional] [CMG](http://www.cs.cmu.edu/~jkoutis/cmg.html) solver (recommended)


## Citation
If you use this code, please cite our papers:
```sh
@inproceedings{peng2017iccvw,
 author =  {Songyou Peng and Bjoern Haefner and Yvain Qu{\'e}au and Daniel Cremers},
 title = {{Depth Super-Resolution Meets Uncalibrated Photometric Stereo}},
 year = {2017},
 booktitle = {IEEE International Conference on Computer Vision (ICCV) Workshop},
}
```
and
```sh
@inproceedings{haefner2018pdsr,
 author =  {Bjoern Haefner and Songyou Peng and Alok Verma and Yvain Qu{\'e}au and Daniel Cremers},
 title = {Photometic Depth Super-Resolution},
 year = {2018},
 booktitle = {arXiv preprint arXiv:1809.10097},
}
```
Contact **Songyou Peng** [:envelope:](mailto:psy920710@gmail.com) for questions, comments and reporting bugs.
