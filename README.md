# Adversarial Reciprocal Points Learning for Open Set Recognition
Official PyTorch implementation of ["**Adversarial Reciprocal Points Learning for Open Set Recognition**"](https://arxiv.org/abs/2103.00953).

<p align="center">
    <img src=./img/ARPL.jpg width="800">
</p>

## 1. Requirements
### Environments
Currently, requires following packages
- python 3.6+
- torch 1.4+
- torchvision 0.5+
- CUDA 10.1+
- scikit-learn 0.22+

### Datasets
For Tiny-ImageNet, please download the following datasets to ```./data/tiny_imagenet```.
-   [tiny_imagenet](https://drive.google.com/file/d/1oJe95WxPqEIWiEo8BI_zwfXDo40tEuYa/view?usp=sharing)

## 2. Training & Evaluation

### Open Set Recognition
To train open set recognition models in paper, run this command:
```train
python osr.py --dataset <DATASET> --loss <LOSS>
```
> Option --loss can be one of ARPLoss/RPLoss/GCPLoss/Softmax. --dataset is one of mnist/svhn/cifar10/cifar100/tiny_imagenet. To run ARPL+CS, add --cs after this command.

### Out-of-Distribution Detection
To train out-of-distribution models in paper, run this command:
```train
python ood.py --dataset <DATASET> --out-dataset <DATASET> --model <NETWORK> --loss <LOSS>
```
> Option --out-dataset denotes the out-of-distribution dataset for evaluation. --loss can be one of ARPLoss/RPLoss/GCPLoss/Softmax. --dataset is one of mnist/cifar10. --out-dataset is one of kmnist/svhn/cifar100. To run ARPL+CS, add --cs after this command.

### Evaluation
To evaluate the trained model for Open Set Classification Rate (OSCR) and Out-of-Distribution (OOD) detection setting, add ```--eval``` after the training command.

## 3. Results
### We visualize the deep feature of Softmax/GCPL/ARPL/ARPL+CS as below.

<p align="center">
    <img src=./img/results.jpg width="800">
</p>

> Colored triangles represent the learned reciprocal points of different known classes.

## 4. PKU-AIR300
<p align="center">
    <img src=./img/thumb.jpg width="600"> 
</p>

A new large-scale challenging aircraft dataset for open set recognition: [Aircraft 300 (Air-300)](https://github.com/iCGY96/ARPL/blob/main/AIR300.md). It contains 320,000 annotated colour images from 300 different classes in total. Each category contains 100 images at least, and a maximum of 10,000 images, which leads to the long tail distribution. 


## Citation
- If you find our work or the code useful, please consider cite our paper using:
```bibtex
@inproceedings{chen2021adversarial,
    title={Adversarial Reciprocal Points Learning for Open Set Recognition},
    author={Chen, Guangyao and Peng, Peixi and Wang, Xiangqian and Tian, Yonghong},
    journal={arXiv preprint arXiv:2103.00953},
    year={2021}
}
```


- All publications using Air-300 Dataset should cite the paper below:
```bibtex
@InProceedings{chen_2020_ECCV,
    author = {Chen, Guangyao and Qiao, Limeng and Shi, Yemin and Peng, Peixi and Li, Jia and Huang, Tiejun and Pu, Shiliang and Tian, Yonghong},
    title = {Learning Open Set Network with Discriminative Reciprocal Points},
    booktitle = {The European Conference on Computer Vision (ECCV)},
    month = {August},
    year = {2020}
}
```