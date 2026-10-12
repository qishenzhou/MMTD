"""Load one fixed-sensor batch: python -m examples.load_fixed_sensor --data-root ..."""

import argparse
from pathlib import Path

from data_provider import MadridFSDataset4NP, NPSampler1, STDataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--cluster-path", type=Path)
    parser.add_argument("--num-cluster", type=int, default=10)
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()
    dataset = MadridFSDataset4NP(
        distance_matrix_path=str(args.data_root / "distance_matrix.h5"),
        static_features_path=str(args.data_root / "road_info.h5"),
        dynamic_features_paths={"sensor": str(args.data_root / "fixed_sensor_data.h5")},
        target_features="q",
        input_features="q",  # retained for compatibility; x includes all sensor features
        input_window=12,
        pred_window=4,
        threshold_distance=1,
        split=args.split,
        not_scale=args.no_scale,
    )
    sampler = NPSampler1(dataset, t_batch_size=2, s_batch_size=1,
                         num_cluster=args.num_cluster, cluster_path=args.cluster_path,
                         shuffle=False)
    loader = STDataLoader(dataset, sampler, num_workers=0)
    if len(loader) == 0:
        raise SystemExit("No windows available; use more timestamps or a different split.")
    batch = next(iter(loader))
    print(batch)
    print("x:", tuple(batch.x.shape), "y:", tuple(batch.y.shape))
    print("Unobserved nodes:", int(batch.label_unobs_mask.sum()))
    print("Target values in original units:", tuple(loader.inverse_transform(batch.y).shape))


if __name__ == "__main__":
    main()
