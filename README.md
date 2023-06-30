# Adversarial Reciprocal Points Learning for Open Set Recognition (TPAMI'21)
Official PyTorch implementation of ["**Adversarial Reciprocal Points Learning for Open Set Recognition (TPAMI'21)**"](https://ieeexplore.ieee.org/document/9521769), [Guangyao Chen](https://scholar.google.com/citations?hl=zh-CN&user=zvHHe1UAAAAJ), [Peixi Peng](https://scholar.google.com/citations?hl=zh-CN&user=CFMuFGoAAAAJ), Xiangqian Wang, and [Yonghong Tian](https://scholar.google.com/citations?hl=zh-CN&user=fn6hJx0AAAAJ).

> **Abstract:** *Open set recognition (OSR), aiming to simultaneously classify the seen classes and identify the unseen classes as 'unknown', is essential for reliable machine learning.The key challenge of OSR is how to reduce the empirical classification risk on the labeled known data and the open space risk on the potential unknown data simultaneously. To handle the challenge, we formulate the open space risk problem from the perspective of multi-class integration, and model the unexploited extra-class space with a novel concept Reciprocal Point. Follow this, a novel learning framework, termed Adversarial Reciprocal Point Learning (ARPL), is proposed to minimize the overlap of known distribution and unknown distributions without loss of known classification accuracy. Specifically, each reciprocal point is learned by the extra-class space with the corresponding known category, and the confrontation among multiple known categories are employed to reduce the empirical classification risk. Then, an adversarial margin constraint is proposed to reduce the open space risk by limiting the latent open space constructed by reciprocal points. To further estimate the unknown distribution from open space, an instantiated adversarial enhancement method is designed to generate diverse and confusing training samples, based on the adversarial mechanism between the reciprocal points and known classes. This can effectively enhance the model distinguishability to the unknown classes. Extensive experimental results on various benchmark datasets indicate that the proposed method is significantly superior to other existing approaches and achieves state-of-the-art performance.*

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
-   [tiny_imagenet](https://drive.google.com/file/d/1vR8ltP_U0UCM42pqz8q4mTbXcvipNNWP/view?usp=sharing)

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

A new large-scale challenging aircraft dataset for open set recognition: [Aircraft 300 (Air-300)](https://github.com/iCGY96/ARPL/blob/master/AIR300.md). It contains 320,000 annotated colour images from 300 different classes in total. Each category contains 100 images at least, and a maximum of 10,000 images, which leads to the long tail distribution. 


## Citation
If you find our work and this repository useful. Please consider giving a star :star: and citation.
```bibtex
@article{chen2021adversarial,
  author={Chen, Guangyao and Peng, Peixi and Wang, Xiangqian and Tian, Yonghong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Adversarial Reciprocal Points Learning for Open Set Recognition}, 
  year={2021},
  doi={10.1109/TPAMI.2021.3106743}
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
