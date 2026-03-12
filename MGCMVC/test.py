
from __future__ import annotations

import argparse
import torch

from dataloader import load_data
from metric import valid
from network import Network


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a saved MGCMVC checkpoint')
    parser.add_argument('--dataset', default='MFeat', choices=['MFeat', 'Reuters', 'AwA2'])
    parser.add_argument('--data_root', default='.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gpu', default='0')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, input_dims, view, data_size, class_num, _ = load_data(args.dataset, data_root=args.data_root)
    low_feature_dim = 128
    high_feature_dim = 256
    dims = [500, 500, 2000]
    model = Network(view, input_dims, low_feature_dim, high_feature_dim, dims, class_num, device).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    valid(model, device, dataset, view, data_size, class_num, eval_h=True, dataset_name=args.dataset, seed=0)


if __name__ == '__main__':
    main()
