import os
import json
import argparse
from dataclasses import asdict

import numpy as np

from DCMVC import Config, train_dcmvc
from eval.cluster_report import cluster_composition_table

from data.load_mfeat import load_mfeat
from data.load_reuters21578 import load_reuters21578_kmeans
from data.load_awa2 import load_awa2


def parse_seeds(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mfeat", "reuters", "awa2"], required=True)
    ap.add_argument("--out", default="eval/outputs")

    # common
    ap.add_argument("--K", type=int, default=None, help="#clusters (if None, inferred when possible)")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")

    # training overrides
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--finetune", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)

    # reuters
    ap.add_argument("--reuters_dir", default="Dataset/Reuters")
    ap.add_argument("--reuters_splits", default="TRAIN")
    ap.add_argument("--top_k_topics", type=int, default=10)
    ap.add_argument("--svd_dim", type=int, default=300)

    # awa2
    ap.add_argument("--awa2_root", default="Dataset/AwA/Animals_with_Attributes2")
    ap.add_argument("--awa2_split", default="all")
    ap.add_argument("--max_per_class", type=int, default=200)

    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)

    # -------- load data --------
    class_names = None

    if args.dataset == "mfeat":
        views, y_true = load_mfeat("Dataset/Handwritten")
        K = 10 if args.K is None else args.K
        class_names = [str(i) for i in range(K)]
        ds_tag = f"mfeat_K{K}"

    elif args.dataset == "reuters":
        splits = tuple([s.strip().upper() for s in args.reuters_splits.split(",") if s.strip()])
        views, y_true, topic_names = load_reuters21578_kmeans(
            reuters_dir=args.reuters_dir,
            splits=splits,
            top_k_topics=args.top_k_topics,
            svd_dim=args.svd_dim,
        )
        K = args.top_k_topics if args.K is None else args.K
        class_names = topic_names
        ds_tag = f"reuters_K{K}_svd{args.svd_dim}_splits{'-'.join(splits)}"

    else:  # awa2
        views, y_true, class_names = load_awa2(
            root=args.awa2_root,
            split=args.awa2_split,
            max_per_class=args.max_per_class,
        )
        K = int(np.max(y_true) + 1) if args.K is None else args.K
        ds_tag = f"awa2_{args.awa2_split}_K{K}_m{args.max_per_class}"

    # -------- output dir --------
    out_dir = os.path.join(args.out, f"dcmvc_{ds_tag}")
    os.makedirs(out_dir, exist_ok=True)

    # -------- run seeds --------
    per_seed = []

    for s in seeds:
        cfg = Config(seed=s)
        if args.warmup is not None:
            cfg.warmup_epochs = args.warmup
        if args.finetune is not None:
            cfg.finetune_epochs = args.finetune
        if args.alpha is not None:
            cfg.alpha = args.alpha
        if args.beta is not None:
            cfg.beta = args.beta
        if args.batch is not None:
            cfg.batch_size = args.batch
        if args.lr is not None:
            cfg.lr = args.lr

        print("\n" + "=" * 80)
        print(f"[Run] dataset={args.dataset} seed={s} K={K}")
        print("=" * 80)

        res = train_dcmvc(views, n_clusters=K, labels=y_true, cfg=cfg)
        y_pred = res["pred"]

        record = {
            "seed": s,
            "ACC": float(res.get("acc", np.nan)),
            "NMI": float(res.get("nmi", np.nan)),
            "ARI": float(res.get("ari", np.nan)),
            "cfg": asdict(cfg),
        }
        per_seed.append(record)

        # save seed outputs
        np.save(os.path.join(out_dir, f"y_pred_seed{s}.npy"), y_pred)
        np.save(os.path.join(out_dir, f"z_seed{s}.npy"), res["z"])

        # cluster composition table
        df, M = cluster_composition_table(y_true, y_pred, class_names=class_names, topk=5)
        df.to_csv(os.path.join(out_dir, f"composition_seed{s}.csv"), index=False)
        np.save(os.path.join(out_dir, f"contingency_seed{s}.npy"), M)

    # aggregate
    metrics = ["ACC", "NMI", "ARI"]
    mean = {m: float(np.mean([d[m] for d in per_seed])) for m in metrics}
    std = {m: float(np.std([d[m] for d in per_seed], ddof=1)) for m in metrics} if len(per_seed) > 1 else {m: 0.0 for m in metrics}

    summary = {"per_seed": per_seed, "mean": mean, "std": std}

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[Done]")
    print("Mean:", mean)
    print("Std :", std)
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
