# Dual Contrast-Driven Deep Multi-View Clustering

This repo contains the code and data associated with our [DCMVC](https://ieeexplore.ieee.org/document/10648641) accepted by **IEEE Transactions on Image Processing 2024**.

## Framework

![Framework Diagram](fig/framework.png)

The overall framework of the proposed DCMVC within an Expectation-Maximization framework. The framework includes: (a) View-specific Autoencoders and Adaptive Feature Fusion Module, which extracts high-level features and fuses them into consensus representations; (b) Dynamic Cluster Diffusion Module, enhancing inter-cluster separation by maximizing the distance between clusters; (c) Reliable Neighbor-guided Positive Alignment Module, improving within-cluster compactness using a pseudo-label and nearest neighbor structure-driven contrastive learning; (d) Clustering-friendly Structure, ensuring well-separated and compact clusters.

## Requirements

hdf5storage==0.1.19

matplotlib==3.5.3

numpy==1.20.1

scikit_learn==0.23.2

scipy==1.7.1

torch==1.8.1+cu111


## Datasets & trained models
The Cora, ALOI-100, Hdigit, and Digit-Product datasets, along with the trained models for these datasets, can be downloaded from [Google Drive](https://drive.google.com/drive/folders/108M1L8fqFk4ZcViZWqQbDe3a2d-uGcXd?usp=drive_link) or [Baidu Cloud](https://pan.baidu.com/s/10vzfz623i4NMx-HslacObQ) password: data.

## Usage

### 0) Prepare your datasets (MFeat / Reuters / AwA2)

This repository's training code expects datasets in `datasets/*.mat` with keys:

- `X`: a MATLAB-style cell array of shape `(1, V)`, where each cell is a float32 array `[N, d_v]`
- `Y`: labels as int32 array `[N, 1]` with values `0..K-1`

Scripts provided:

```bash
python prepare_mfeat.py --data_dir <folder_with_mfeat-files> --out datasets/MFeat.mat
python prepare_reuters.py --data_dir <folder_with_reut2-xxx.sgm> --out datasets/Reuters.mat --top_k 10
python prepare_awa2.py --view1 <view1.npy> --view2 <view2.npy> --labels <labels.npy> --out datasets/AwA2.mat
```

Then train with:

```bash
python train.py --dataset MFeat
python train.py --dataset Reuters
python train.py --dataset AwA2
```

You can override hyperparameters:

```bash
python train.py --dataset MFeat --alpha 0.01 --beta 0.1 --k 10
```

Train a new model：

````python
python train.py
````

Test the trained model:

````python
python test.py
````

## Acknowledgments

Work&Code takes inspiration from [MFLVC](https://github.com/SubmissionsIn/MFLVC), [ProPos](https://github.com/Hzzone/ProPos).

## Citation

If you find our work beneficial to your research, please consider citing:

````latex
@ARTICLE{10648641,
  author={Cui, Jinrong and Li, Yuting and Huang, Han and Wen, Jie},
  journal={IEEE Transactions on Image Processing}, 
  title={Dual Contrast-Driven Deep Multi-View Clustering}, 
  year={2024},
  volume={33},
  number={},
  pages={4753-4764},
  keywords={Feature extraction;Contrastive learning;Reliability;Clustering methods;Task analysis;Data mining;Unsupervised learning;Multi-view clustering;deep clustering;representation learning;contrastive learning},
  doi={10.1109/TIP.2024.3444269}}
````



