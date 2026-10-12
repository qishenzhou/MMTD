"""End-to-end loader checks using generated inputs; no traffic data are distributed."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
import torch

from data_provider import MadridDataset, MadridFSDataset4NP, NPSampler1, STDataLoader


STATIC_NAMES = [
    "Oneway", "RoadType", "Length", "Tunnel", "SpeedLimit", "InDegree",
    "OutDegree", "BetweennessCentrality", "ClosenessCentrality", "Heading",
    "Curviness", "CurvatureRatio", "x", "y", "poi_count", "poi_density",
    "work_count", "work_density", "residential_count", "residential_density",
    "education_count", "education_density", "medical_count", "medical_density",
    "food_count", "food_density", "shopping_count", "shopping_density",
    "entertainment_count", "entertainment_density", "transport_count",
    "transport_density", "other_count", "other_density", "work_residential_ratio",
    "commercial_residential_ratio", "entertainment_residential_ratio",
    "poi_diversity", "poi_mix_entropy", "weekday_dominance", "peak_hour_impact",
    "all_day_impact", "weekend_impact",
]


class LoaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.nodes = np.arange(100, 112)
        self.times = np.array(pd.date_range("2024-08-01", periods=60, freq="15min")
                              .strftime("%Y-%m-%d %H:%M:%S"), dtype="S19")
        self.write("distance_matrix", np.abs(np.arange(12)[:, None] - np.arange(12)) / 10,
                   self.nodes)
        static = np.ones((12, len(STATIC_NAMES)))
        static[:, 4] = np.arange(40, 52)
        self.write("road_info", static, self.nodes, STATIC_NAMES)
        sensor = np.ones((60, 8, 3)) * 50
        sensor[:, :, 0] += np.arange(60)[:, None]
        sensor[1, 0, 0] = np.nan
        self.write("fixed_sensor_data", sensor, self.nodes[:8],
                   ["volume", "occ", "conges_levl"], self.times)

    def write(self, name, data, nodes, features=None, times=None):
        with h5py.File(self.root / f"{name}.h5", "w") as f:
            f["data/values"] = data
            f["metadata/nodes"] = nodes
            if features is not None:
                f["metadata/features"] = np.array(features, dtype="S")
            if times is not None:
                f["metadata/times"] = times

    def dataset(self, cls=MadridFSDataset4NP, tomtom=False, **kwargs):
        paths = {"sensor": str(self.root / "fixed_sensor_data.h5")}
        if tomtom:
            paths["tomtom"] = str(self.root / "tomtom_data.h5")
        return cls(distance_matrix_path=str(self.root / "distance_matrix.h5"),
                   static_features_path=str(self.root / "road_info.h5"),
                   dynamic_features_paths=paths, input_window=3, pred_window=2,
                   target_features="q", input_features="q", permanent_unobs_ratio=0.25,
                   **kwargs)

    def loader(self, dataset, **kwargs):
        return STDataLoader(dataset, NPSampler1(dataset, 2, 1, shuffle=False,
                                                num_cluster=1, **kwargs))

    def test_fixed_sensor_splits_masks_and_scaling(self):
        for split in ("train", "val", "test"):
            with self.subTest(split=split):
                dataset = self.dataset(split=split)
                batch = next(iter(self.loader(dataset)))
                self.assertEqual(batch.x.shape[1:], (5, 3))
                self.assertEqual(batch.y.shape[1:], (5, 1))
                self.assertTrue(torch.isnan(batch.x[:, 3:]).all())
                self.assertTrue(torch.isnan(batch.x[batch.label_unobs_mask]).all())
                self.assertNotIn("dynamic_aux", batch)
                torch.testing.assert_close(dataset.forward_transform(dataset.inverse_transform(batch.y)),
                                           batch.y, equal_nan=True)
                self.assertEqual(dataset.get_metadata()["total_windows"], len(dataset))
                if split != "test":
                    self.assertFalse(np.isin(dataset.full_graph.n_idx,
                                            dataset.permanent_unobs_nodes_global_idx).any())

    def test_unscaled_full_graph_preserves_missing_nodes(self):
        dataset = self.dataset(cls=MadridDataset, split="test", not_scale=True)
        batch = next(iter(self.loader(dataset)))
        self.assertEqual(batch.num_nodes, 24)
        self.assertEqual(batch.y.shape[0], 16)
        self.assertTrue(torch.isnan(batch.x[~batch.graph_mask]).all())
        self.assertEqual(dataset.scale, 1)
        torch.testing.assert_close(dataset.inverse_transform(batch.y), batch.y)

    def test_optional_tomtom_with_partial_reordered_coverage(self):
        nodes = self.nodes[[6, 0, 2, 4]]
        values = np.ones((60, 4, 6), dtype=np.float32) * 20
        self.write("tomtom_data", values, nodes,
                   ["AvgTt", "MedTt", "Hits", "SdSp", "AvgSp", "MedSp"], self.times)
        for no_scale in (False, True):
            dataset = self.dataset(split="test", tomtom=True, not_scale=no_scale)
            batch = next(iter(self.loader(dataset)))
            self.assertEqual(batch.dynamic_aux.shape, (16, 5, 4))
            self.assertEqual(batch.tom_hits.shape, (16, 5, 1))
            missing = ~torch.isin(batch.n_id, torch.from_numpy(nodes))
            self.assertTrue(torch.isnan(batch.dynamic_aux[missing]).all())
            self.assertTrue(torch.isnan(batch.tom_hits[missing]).all())
            if no_scale:
                expected = torch.tensor(20 / (nodes - 100 + 40), dtype=torch.float32)
                np.testing.assert_allclose(dataset.dynamic_info["tomtom"]["data"][0, :, 4], expected)

    def test_external_clusters_and_empty_windows(self):
        cluster_path = self.root / "clusters.csv"
        pd.DataFrame({"node_id": self.nodes, "cluster_id_road_graph": np.arange(12) // 4})\
            .to_csv(cluster_path, index=False)
        dataset = self.dataset(split="test", not_scale=True)
        loader = self.loader(dataset, cluster_path=cluster_path)
        self.assertEqual(loader.num_s_batches, 2)
        self.assertEqual(len(list(loader)), len(loader))
        dataset.valid_indices = np.array([], dtype=int)
        empty = self.loader(dataset, drop_last=True)
        self.assertEqual(list(empty), [])

    def test_sensor_row_order_keeps_graph_edges_aligned(self):
        with h5py.File(self.root / "fixed_sensor_data.h5", "r+") as f:
            f["metadata/nodes"][:] = f["metadata/nodes"][:][::-1]
            f["data/values"][:] = f["data/values"][:][:, ::-1]
        for split in ("train", "test"):
            dataset = self.dataset(split=split, not_scale=True, threshold_distance=0.11)
            graph = dataset.full_graph
            a, b = graph.n_id[graph.edge_index]
            self.assertTrue((torch.abs(a - b) <= 1).all())
            batch = next(iter(self.loader(dataset)))
            self.assertTrue(torch.equal(batch.n_id[:graph.num_nodes], graph.n_id)
                            if split == "test" else batch.num_nodes > 0)


if __name__ == "__main__":
    unittest.main()
