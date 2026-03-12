
# MGCMVC Adapted for MFeat / Reuters / AwA2

This repo is a light adaptation of the original MGCMVC code so it can run on:
- `MFeat` (UCI Multiple Features / Handwritten)
- `Reuters` (top-10 single-topic TRAIN split, 2 text views)
- `AwA2` (ResNet18 image features + class-level attributes)

## Key changes
- Removed hardcoded original datasets and added loaders for the three dissertation datasets.
- Added `--seed`, `--data_root`, `--gpu`, and output directory control.
- Saves predictions and fused embeddings to `outputs/preds/` for later diagnostics.
- Saves per-epoch metrics history to CSV and best metrics to JSON.
- Model saving is optional via `--save_model`.

## Install
```bash
pip install -r requirements.txt
```

## Run
From the repo root:
```bash
python train.py --dataset MFeat --data_root /path/to/project --gpu 0 --seed 0
python train.py --dataset Reuters --data_root /path/to/project --gpu 0 --seed 0
python train.py --dataset AwA2 --data_root /path/to/project --gpu 0 --seed 0
```

`data_root` should be the folder that contains the dissertation project structure, e.g.:
- `Dataset/Handwritten/mfeat-*`
- `Dataset/Reuters/reut2-*.sgm`
- `Dataset/AwA/Animals_with_Attributes2/...`

## Outputs
- `outputs/<DATASET>_seed<SEED>_history.csv`
- `outputs/<DATASET>_seed<SEED>_best.json`
- `outputs/preds/<DATASET>_seed<SEED>_ypred.npy`
- `outputs/preds/<DATASET>_seed<SEED>_H.npy`
- `outputs/preds/<DATASET>_seed<SEED>_ytrue.npy`

## Notes
- Reuters loader matches the dissertation KMeans preprocessing more closely than the original MGCMVC repo.
- AwA2 uses image-derived ResNet18 features plus repeated semantic attributes, so results should be described as a multimodal feature setting rather than raw-image-only clustering.
