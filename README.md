# SRmeetsPS

This repository contains the code for our paper:  
Songyou Peng, Bjoern Haefner, Yvain Queau and Daniel Cremers, "**[Depth Super-Resolution Meets Uncalibrated Photometric Stereo](https://arxiv.org/abs/1708.00411)**", In IEEE Conference on Computer Vision (ICCV) Workshop, 2017.

A CUDA version code is also available [here](https://github.com/nihalsid/SRmeetsPS-CUDA).

## Input
* Super-resolution RGB images (at least **4** images)
* Super-resolution binary mask
* Low-resolution depth images (1 image is fine, same size as RGB image is also fine)
* Intrinsic matrix (containing the focal length and principle points of the RGB images)
* Downsampling matrix (you can aquire with ``getDownsampleMat.m``. The file name should be like ``D_1280_960_2.mat``)

You can refer to the examples in the ``Data`` folder.

## Requirement
* MATLAB (tested and working in R2015b and later versions)
* [Optional] [CMG](http://www.cs.cmu.edu/~jkoutis/cmg.html) solver (recommended)


## Citation
If you use this code, please cite our paper:
```sh
@inproceedings{peng2017iccvw,
 author =  {Songyou Peng and Bjoern Haefner and Yvain Qu{\'e}au and Daniel Cremers},
 title = {{Depth Super-Resolution Meets Uncalibrated Photometric Stereo}},
 year = {2017},
 booktitle = {IEEE International Conference on Computer Vision (ICCV) Workshop},
}
```
Contact **Songyou Peng** [:envelope:](mailto:psy920710@gmail.com) for questions, comments and reporting bugs.
