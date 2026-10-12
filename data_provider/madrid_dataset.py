"""Dataset adapted from TA-ANP for prepared MMTD HDF5 inputs."""

from typing import List, Tuple
import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import subgraph
from .time_features_tool import time_features
from datetime import datetime, timezone


class MadridDataset(Dataset):
    """
    A dataset for Madrid network data using PyTorch Geometric format.

    This dataset handles:
    - Static graph structure (distance matrix)
    - Dynamic node features (multiple sources with missing data)
    - Static node features
    """

    def __init__(
            self,
            distance_matrix_path: str,
            static_features_path: str,
            target_features: str,
            input_features: str,
            dynamic_features_paths: dict[str, str],
            input_window=12,
            look_back_window=0,
            pred_window=12,
            stride=1,
            split="train",
            train_ratio=0.6,
            val_ratio=0.2,
            train_unobs_ratio=0.05,
            permanent_unobs_ratio=0.1,
            threshold_distance=10,  # Distance threshold for adjacency matrix (km)
            seed4unob=42,
            not_scale=False,
            stack_time=False,
            add_node_list: list[int] = None,
            inf_exclude_num: int = 0,
            add_by_betweenness: int = 0,
            add_by_closeness: int = 0):
        """
        Initialize the MadridDataset.

        Args:
            distance_matrix_path (str): Path to distance matrix HDF5 file
            static_features_path (str): Path to static features HDF5 file
            dynamic_features_paths (dict[str, str]): Dictionary of paths to dynamic features HDF5 files
            input_window_size (int, optional): Size of the input temporal window. Defaults to 12.
            look_back_window_size (int, optional): Size of the loop back temporal window. Defaults to 0.
            pred_window_size (int, optional): Size of the output temporal window for prediction. Defaults to 12.
            stride (int, optional): Stride for sliding window. Defaults to 1.
            split (str, optional): 'train', 'val', or 'test'. Defaults to "train".
            train_ratio (float, optional): Ratio of training data. Defaults to 0.6.
            val_ratio (float, optional): Ratio of validation data. Defaults to 0.2.
            train_unobs_ratio (float, optional): Ratio of unobserved nodes in training data. Defaults to 0.05.
            permanent_unobs_ratio (float, optional): Ratio of unobserved nodes in permanent nodes. Defaults to 0.1.
            threshold_distance (int, optional): Threshold to create adjacency matrix from distance matrix. Defaults to 10.
            not_scale (bool, optional): Whether to scale the data. Defaults to False.
            stack_time (bool, optional): Whether to stack the time step dimension. Defaults to False.
            add_node_list (list[int], optional): List of add node ids. Defaults to None.
            inf_exclude_num (int, optional): Number of nodes to exclude for inference. Defaults to 0.
            add_by_betweenness (int, optional): Number of nodes to add by betweenness. Defaults to 0.
            add_by_closeness (int, optional): Number of nodes to add by closeness. Defaults to 0.
        """
        assert look_back_window >= 0 and look_back_window <= input_window, f"look_back_window_size {look_back_window} must be between 0 and {input_window}"
        super().__init__(log=False)

        self.distance_matrix_path = distance_matrix_path
        self.static_features_path = static_features_path
        self.dynamic_features_paths = dynamic_features_paths
        self.add_node_list = add_node_list
        self.inf_exclude_num = inf_exclude_num
        self.add_by_betweenness = add_by_betweenness
        self.add_by_closeness = add_by_closeness
        fea_dict = {
            'q': [0],
            'o': [1],
            'c': [2],
        }
        self.tar_feas = fea_dict[target_features]
        self.input_window = input_window
        self.look_back_window = look_back_window
        self.pred_window = pred_window
        self.total_window_size = input_window + pred_window
        self.stride = stride

        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.train_unobs_ratio = train_unobs_ratio
        self.permanent_unobs_ratio = permanent_unobs_ratio
        self.seed4unobgen = seed4unob

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if input_window < 1 or pred_window < 0 or stride < 1:
            raise ValueError("input_window and stride must be positive; pred_window must be nonnegative")
        if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 - train_ratio):
            raise ValueError("train/validation ratios must leave a nonempty test fraction")
        if not 0 <= permanent_unobs_ratio < 1:
            raise ValueError("permanent_unobs_ratio must be in [0, 1)")
        if "sensor" not in dynamic_features_paths or set(dynamic_features_paths) - {"sensor", "tomtom"}:
            raise ValueError("Provide sensor data and, optionally, already aligned tomtom data")
        self.threshold_distance = threshold_distance
        self.not_scale = not_scale
        self.stack_time = stack_time
        # Initialize data structures
        self._init_data()

        self._scale_data()

        self.global_node_ids = torch.tensor(self.global_node_ids).long()
        self.full_graph = Data(
            edge_index=self.full_edge_index,
            edge_attr=self.full_weights,
            num_nodes=len(self.train_node_global_idx)
            if self.split != "test" else len(self.global_node_ids),
            n_idx=self.train_node_global_idx if self.split != "test" else torch.tensor(
                range(len(self.global_node_ids))),
            n_id=self.global_node_ids[self.train_node_global_idx]
            if self.split != "test" else self.global_node_ids,
        )

        self.causal_time_edges(self.input_window, self.pred_window + self.input_window)

    def _init_data(self):
        """
        Initialize dataset-specific data structures.

        This method:
        1. Builds or loads the graph structure
        2. Creates node ID mappings
        3. Identifies active and unobserved nodes
        4. Loads dynamic data information
        5. Sets up dataset splits
        """

        # Get all node information and create mappings
        with h5py.File(self.distance_matrix_path, "r") as f:
            self.global_node_ids = f["metadata/nodes"][:].astype(np.int64)

        # Create bidirectional mappings between physical IDs and continuous indices
        self.node_id_to_global_idx = {
            node_id: idx
            for idx, node_id in enumerate(self.global_node_ids)
        }
        self.global_idx_to_node_id = {
            idx: node_id
            for node_id, idx in self.node_id_to_global_idx.items()
        }

        # Set dataset size
        with h5py.File(self.dynamic_features_paths["sensor"], "r") as f:
            self.timestamps = f["metadata/times"][:]
            self.timestamps = pd.to_datetime(
                [ts.decode('utf-8') for ts in self.timestamps])
            self.num_timestamps = len(self.timestamps)
        self.train_start_idx = 0
        self.train_end_idx = int(self.num_timestamps *
                                 self.train_ratio)  # we keep this for scale data
        # set dataset split
        if self.split == "train":
            self.start_idx = 0
            self.end_idx = int(self.num_timestamps * self.train_ratio)
        elif self.split == "val":
            self.start_idx = int(self.num_timestamps * self.train_ratio)
            self.end_idx = int(self.num_timestamps * (self.train_ratio + self.val_ratio))
        else:
            self.start_idx = int(self.num_timestamps *
                                 (self.train_ratio + self.val_ratio))
            self.end_idx = self.num_timestamps

        self.split_timestamps = self.timestamps[self.start_idx:self.end_idx]
        self.valid_indices = []
        for i in range(
                0,
                len(self.split_timestamps) - self.input_window - self.pred_window + 1,
                self.stride):
            self.valid_indices.append(i)
        self.valid_indices = np.array(self.valid_indices)
        # Build or load the graph structure
        self._prepare_graph_data()
        # Get active nodes (nodes with fixed sensors)
        # and split active nodes into permanent unobserved and observable nodes
        with h5py.File(self.dynamic_features_paths["sensor"], "r") as f:
            self.active_node_ids = f["metadata/nodes"][:]
            all_active_node_global_idx = np.array(
                [self.node_id_to_global_idx[nid] for nid in self.active_node_ids])
            num_permanent_unobs = int(
                len(all_active_node_global_idx) * self.permanent_unobs_ratio)
            np.random.seed(self.seed4unobgen)
            permanent_unobs_local_idx = np.random.choice(
                len(all_active_node_global_idx), size=num_permanent_unobs,
                replace=False)  # used to slect nodes in all_active_node_global_idx
            # For training phase: remove permanent unobserved nodes from active nodes
            mask = np.ones(len(all_active_node_global_idx), dtype=bool)
            mask[permanent_unobs_local_idx] = False
            if self.add_node_list is not None:
                mask[np.isin(
                    self.active_node_ids, np.array(self.add_node_list)
                )] = True  # some permanently unobserved nodes will be added manually, for the experiment of uncertainty-guided active sensing
            elif self.inf_exclude_num > 0:  # for the experiment of inference with addition unobs nodes, to see if uncertainty will be higher or not
                # Randomly select num_inf_exclude nodes with mask=True and set their mask to False.
                inf_exclude_local_idx = np.random.choice(np.where(mask)[0],
                                                         size=self.inf_exclude_num,
                                                         replace=False)
                mask[inf_exclude_local_idx] = False
            elif self.add_by_betweenness > 0:
                with h5py.File(self.static_features_path, "r") as f:
                    self.static_features = f["data/values"][:]
                    self.static_features_name = [
                        f.decode() for f in f["metadata/features"][:]
                    ]
                betweenness = self.static_features[:,
                                                   self.static_features_name.
                                                   index("BetweennessCentrality")]
                unobs_now_global_idx = all_active_node_global_idx[~mask]
                unobs_now_betweenness = betweenness[unobs_now_global_idx]
                add_unobs_global_idx = unobs_now_global_idx[np.argsort(
                    unobs_now_betweenness)[-self.add_by_betweenness:]]
                mask[np.isin(all_active_node_global_idx, add_unobs_global_idx)] = True
            elif self.add_by_closeness > 0:
                with h5py.File(self.static_features_path, "r") as f:
                    self.static_features = f["data/values"][:]
                    self.static_features_name = [
                        f.decode() for f in f["metadata/features"][:]
                    ]
                closeness = self.static_features[:,
                                                 self.static_features_name.
                                                 index("ClosenessCentrality")]
                unobs_now_global_idx = all_active_node_global_idx[~mask]
                unobs_now_closeness = closeness[unobs_now_global_idx]
                add_unobs_global_idx = unobs_now_global_idx[np.argsort(
                    unobs_now_closeness)[-self.add_by_closeness:]]
                mask[np.isin(all_active_node_global_idx, add_unobs_global_idx)] = True
            self.permanent_unobs_nodes_global_idx = all_active_node_global_idx[~mask]

        # For training phase: remove permanent unobserved nodes from active nodes
        # and filter out edges that related to permanent unobserved nodes
        if self.split != "test":
            self.active_node_global_idx = all_active_node_global_idx[mask]
            self.train_node_global_idx = torch.tensor(
                list(
                    set(self.global_idx_to_node_id.keys()) -
                    set(self.permanent_unobs_nodes_global_idx)))
            self.full_edge_index, self.full_weights = subgraph(self.train_node_global_idx,
                                                               self.full_edge_index,
                                                               self.full_weights,
                                                               relabel_nodes=True)
        else:
            # For test phase: use permanent unobserved nodes as inference targets
            self.active_node_global_idx = all_active_node_global_idx

        # Get static features
        with h5py.File(self.static_features_path, "r") as f:
            self.static_features = f["data/values"][:]
            self.static_features_name = [f.decode() for f in f["metadata/features"][:]]

        # Get dynamic data information
        self.dynamic_info = {}
        for data_source, path in self.dynamic_features_paths.items():
            with h5py.File(path, "r") as f:
                data = f["data/values"][self.start_idx:self.end_idx]
                if data_source == "sensor":
                    # type of data is numpy.ndarray, shape is (T,N,F), we need to replace the row with nan if the first feature is bigger than 300
                    data[data[:, :, 0] > 300, :] = np.nan
                    data[data[:, :, 0] < 10, :] = np.nan
                    # if self.tar_feas == [2]:  # congestion level
                    #     data[data[..., -1] <= 5, :] = np.nan
                elif data_source == "tomtom":
                    # type of data is numpy.ndarray, shape is (T,N,F), we need to replace the row with nan if the any of features is bigger than 120
                    data[(data[:, :, 4:] > 120).any(axis=2), :] = np.nan

                times = f["metadata/times"][self.start_idx:self.end_idx]
                features_name = [f.decode() for f in f["metadata/features"][:]]
                source_node_ids = f["metadata/nodes"][:]

                # Create mask and mapping for this data source
                has_feature_mask = np.zeros(len(self.global_node_ids), dtype=bool)
                global_idx_to_source_idx = np.full(
                    len(self.global_node_ids),
                    np.nan)  # np.nan denotes there is no source in this global ind
                source_node_id_to_source_idx = {
                    nid: idx
                    for idx, nid in enumerate(source_node_ids)
                }

                # create a mask for nodes that have source data and a mapping from global index to source index
                for global_idx, node_id in enumerate(self.global_node_ids):
                    if node_id in source_node_id_to_source_idx:
                        has_feature_mask[global_idx] = True
                        global_idx_to_source_idx[global_idx] = (
                            source_node_id_to_source_idx[node_id])

                # For training phase, mask out permanent unobserved nodes in sensor data
                if self.split != "test" and data_source == "sensor":
                    has_feature_mask[self.permanent_unobs_nodes_global_idx] = False
                    global_idx_to_source_idx[
                        self.permanent_unobs_nodes_global_idx] = np.nan
                if data_source == "tomtom":
                    # norm speed with speed limit
                    source_indices = [self.node_id_to_global_idx[nid] for nid in source_node_ids]
                    self._tom_speed_limits = self.static_features[source_indices, 4].copy()
                    data[..., 4:] /= self._tom_speed_limits[np.newaxis, :, np.newaxis]
                self.dynamic_info[data_source] = {
                    "data": data,
                    "times": times,
                    "features_name": features_name,
                    "has_feature_mask": has_feature_mask,
                    "global_idx_to_source_idx": global_idx_to_source_idx,
                }

    def _prepare_graph_data(self) -> None:
        """Build distance-threshold edges in row chunks (distances are in km)."""
        edges, values = [], []
        with h5py.File(self.distance_matrix_path, "r") as f:
            matrix = f["data/values"]
            for start in range(0, len(self.global_node_ids), 256):
                distances = matrix[start:start + 256]
                rows, cols = np.where(np.isfinite(distances) &
                                      (distances <= self.threshold_distance) &
                                      (distances >= 0))
                edges.append(np.stack([rows + start, cols]))
                values.append(distances[rows, cols])
        distances = np.concatenate(values)
        if distances.size == 0:
            raise ValueError("No valid edges found in the graph")
        sigma = distances.std()
        weights = (np.exp(-(distances ** 2) / (2 * sigma ** 2))
                   if sigma > 0 else np.ones_like(distances))
        self.full_edge_index = torch.from_numpy(np.concatenate(edges, axis=1)).long()
        self.full_weights = torch.from_numpy(weights).float()


    def causal_time_edges(self, t_k, t_q):
        """
        construct causal time edges
        Args:
            t_k: Number of historical time steps (T_k = 0, 1, ..., t_k-1)
            t_q: Number of query time steps (T_q = 0, 1, ..., t_q-1), where t_q >= t_k

        Returns:
            time_edge_index: [2, E_t] Temporal edge indices
        """
        edges = []
        assert t_q >= t_k
        for t_q_idx in range(t_q):
            max_t_k_idx = min(t_q_idx, t_k - 1)

            for t_k_idx in range(max_t_k_idx + 1):
                edges.append([t_q_idx, t_k_idx])
        time_edge_index = torch.tensor(edges).T  # [2, E_t]
        self.causal_time_edge_index = time_edge_index

        # Also construct temporal edge indices without causal constraints.
        self.full_time_edge_index = torch.tensor([[t_q_idx, t_k_idx]
                                                  for t_q_idx in range(t_q)
                                                  for t_k_idx in range(t_q)],
                                                 dtype=torch.long)

    def _scale_data(self):
        """Fit dynamic scalers on the training period; retain NaNs as missing data."""
        self.dynamic_scaler = {}
        if not self.not_scale:
            names = ["Length", "SpeedLimit", "BetweennessCentrality",
                     "ClosenessCentrality", "Heading", "x", "y", "poi_density",
                     "work_density", "residential_density", "education_density",
                     "medical_density", "food_density", "shopping_density",
                     "entertainment_density", "transport_density", "other_density",
                     "poi_diversity", "poi_mix_entropy", "Curviness", "CurvatureRatio"]
            indices = [self.static_features_name.index(name) for name in names]
            self.static_features[:, indices] = StandardScaler().fit_transform(
                self.static_features[:, indices])
            for source, path in self.dynamic_features_paths.items():
                with h5py.File(path, "r") as f:
                    training = f["data/values"][:self.train_end_idx].astype(np.float32)
                    node_ids = f["metadata/nodes"][:]
                if source == "sensor":
                    training[(training[..., 0] > 300) | (training[..., 0] < 10)] = np.nan
                    held_out = self.global_node_ids[self.permanent_unobs_nodes_global_idx]
                    training[:, np.isin(node_ids, held_out)] = np.nan
                elif source == "tomtom":
                    training[(training[..., 4:] > 120).any(axis=2)] = np.nan
                    training[..., 4:] /= self._tom_speed_limits[None, :, None]
                scaler = StandardScaler().fit(training.reshape(-1, training.shape[-1]))
                self.dynamic_scaler[source] = scaler
                values = self.dynamic_info[source]["data"]
                self.dynamic_info[source]["data"] = scaler.transform(
                    values.reshape(-1, values.shape[-1])).reshape(values.shape)
        self.static_features = torch.as_tensor(self.static_features, dtype=torch.float32)
        for info in self.dynamic_info.values():
            info["data"] = torch.as_tensor(info["data"], dtype=torch.float32)


    def inverse_transform(self, y):
        if not self.not_scale:
            y = y * self.scale + self.center
        return y

    def forward_transform(self, y):
        if not self.not_scale:
            y = (y - self.center) / self.scale
        return y

    def tom_inverse_transform(self, y):
        if not self.not_scale:
            y = y * self.tom_scale + self.tom_center
        return y

    def tom_forward_transform(self, y):
        if not self.not_scale:
            y = (y - self.tom_center) / self.tom_scale
        return y

    def len(self):
        """
        Return the number of examples in the dataset.

        Returns:
            int: Number of temporal windows in the dataset
        """
        return max(0, len(self.valid_indices))

    def __getitem__(
        self,
        idx: Tuple[int, List[int]],
    ):
        """
        Extend __getitem__ method to support spatial and temporal indexing.

        Args:
            idx (Tuple[int, List[int]]): A tuple containing:
                - idx[0] (int): temporal index
                - idx[1] (Data): pyg data object
        Returns:
            Data: A PyG data object containing the requested spatio-temporal data
        """
        data = self.get(self.indices()[idx[0]], idx[1])
        return data

    def get(self, t_idx: int, spatial_batch: Data) -> Data:
        """ method to get part of data from the dataset
        Args:
            t_idx (int): temporal index
            spatial_batch (Data): spatial indexes for unobserved nodes

        Returns:
            Data: A PyG data object containing:
                - x: Node features from fixed sensors
                - y: Target features within input and output window
                - edge_index: Edge connectivity
                - edge_attr: Edge weights
                - pos: Node positions
                - time: Temporal features
                - static_aux: Static auxiliary features
                - dynamic_aux: Dynamic auxiliary features
                - label_obs_mask: Mask for observed labels
                - label_unobs_mask: Mask for unobserved labels
                - graph_mask: Overall graph mask
                - num_nodes: Number of nodes in the subgraph
        """
        # Convert relative index to global index
        input_start_idx = self.valid_indices[t_idx]
        input_end_idx = input_start_idx + self.input_window
        output_start_idx = input_end_idx - self.look_back_window
        output_end_idx = input_end_idx + self.pred_window
        all_window = self.input_window + self.pred_window
        static_features = self.static_features[spatial_batch.n_idx].unsqueeze(1).repeat(
            1, all_window, 1)

        input_dynamic_features = {}
        for data_source in self.dynamic_info.keys():
            # inputs
            # Get valid input nodes and their mappings
            if data_source == "sensor":
                global_has_feature_mask = self.dynamic_info[data_source][
                    "has_feature_mask"].copy()
                global_has_feature_mask[spatial_batch.unobs_global_idx] = False
            else:
                global_has_feature_mask = self.dynamic_info[data_source][
                    "has_feature_mask"]  # tomtom is avaiable for these positions.

            subgraph_input_mask = np.atleast_1d(
                global_has_feature_mask[spatial_batch.n_idx])
            subgraph_source_input_idx = np.atleast_1d(
                self.dynamic_info[data_source]["global_idx_to_source_idx"][
                    spatial_batch.n_idx])[subgraph_input_mask]

            inputs = (torch.zeros((
                all_window,
                spatial_batch.num_nodes,
                len(self.dynamic_info[data_source]["features_name"]),
            )) * torch.nan)
            inputs[:self.input_window,
                   subgraph_input_mask] = self.dynamic_info[data_source]["data"][
                       input_start_idx:input_end_idx, subgraph_source_input_idx.astype(int)]
            inputs = inputs.permute(1, 0, 2)  # [N, T, C]

            input_dynamic_features.update({
                data_source: inputs,
            })
            if data_source == "tomtom":
                hits = torch.full((all_window, spatial_batch.num_nodes, 1), torch.nan)
                hits[:, subgraph_input_mask] = self.dynamic_info[data_source]["data"][
                    input_start_idx:output_end_idx, subgraph_source_input_idx.astype(int), 2:3]
                hits = hits.permute(1, 0, 2)  # [N, T, C]
                input_dynamic_features.update({
                    'tomtom_hits': hits,
                })

            # labels
            if data_source == "sensor":
                subgraph_label_mask = np.atleast_1d(
                    self.dynamic_info[data_source]["has_feature_mask"][
                        spatial_batch.n_idx])
                subgraph_source_label_idx = np.atleast_1d(
                    self.dynamic_info[data_source]["global_idx_to_source_idx"][
                        spatial_batch.n_idx])[subgraph_label_mask]

                labels = self.dynamic_info[data_source]["data"][
                    input_start_idx:output_end_idx, subgraph_source_label_idx.astype(int),
                    self.tar_feas].permute(1,
                                           0).unsqueeze(-1)  # [N_sensor, T_in+T_out, C]
                subgraph_label_mask = torch.from_numpy(subgraph_label_mask).bool()
                subgraph_label_unobs_mask = torch.isin(spatial_batch.n_idx,
                                                       spatial_batch.unobs_global_idx)

                # time features
                time = self.dynamic_info[data_source]["times"][
                    input_start_idx:output_end_idx]
                decode_vec = np.vectorize(lambda x: x.decode())
                time = decode_vec(time)
                time_enc = time_features(time, freq="min")
                time_enc = (torch.from_numpy(time_enc).long().unsqueeze(0)
                            )  # Indicates a time stamp series for one graph

                # time id
                time_id = self.dynamic_info[data_source]["times"][
                    output_start_idx].decode()
                time_id = int(
                    datetime.strptime(
                        time_id,
                        "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
        # Create PyG data object
        data = Data(
            x=input_dynamic_features[
                "sensor"],  # Nodes monitored by fixed sensor[no unobserved nodes]
            y=labels,  # Target features: sensor data within output window
            tom_hits=input_dynamic_features.get("tomtom_hits"),
            edge_index=spatial_batch.edge_index,  # Edge connectivity
            edge_attr=spatial_batch.edge_attr,  # Edge weights
            pos=static_features[..., 12:14],
            # spectral_features=self.fourier_features[spatial_batch.n_idx],
            t_id=torch.tensor([time_id] * spatial_batch.num_nodes,
                              dtype=torch.int64).reshape(1, -1),
            n_id=spatial_batch.n_id,
            time_enc=time_enc.repeat(spatial_batch.num_nodes, 1, 1),
            static_aux=static_features[..., :12],
            poi_aux=static_features[..., [15, 17, 19, 21, 23, 25, 27, 29, 31, 33]],
            dynamic_aux=(input_dynamic_features["tomtom"][..., 2:]
                         if "tomtom" in input_dynamic_features else None),  # first two features [AvgTt, MedTt] are dropped
            label_unobs_mask=subgraph_label_unobs_mask,
            graph_mask=subgraph_label_mask,
            num_nodes=spatial_batch.num_nodes,
        )
        return data

    @property
    def scale(self):
        # NOTE: index with [self.tar_feas[0]] (a scalar), NOT [self.tar_feas]
        # (a length-1 list). The list form returns a float64 numpy *array*,
        # which promotes float32 tensors to float64 under `y * scale`,
        # doubling memory (OOM) and breaking torch.quantile's dtype check.
        return 1.0 if self.not_scale else self.dynamic_scaler["sensor"].scale_[self.tar_feas[0]]

    @property
    def center(self):
        return 0.0 if self.not_scale else self.dynamic_scaler["sensor"].mean_[self.tar_feas[0]]

    @property
    def tom_scale(self):
        return 1.0 if self.not_scale else self.dynamic_scaler["tomtom"].scale_[2]

    @property
    def tom_center(self):
        return 0.0 if self.not_scale else self.dynamic_scaler["tomtom"].mean_[2]

    def get_metadata(self):
        """
        Get dataset metadata.

        Returns:
            dict: Dictionary containing:
                - num_nodes: Number of active nodes
                - num_timestamps: Total number of timestamps
                - window_size: Size of temporal window
                - stride: Window stride
                - total_windows: Total number of windows
                - global_node_ids: List of all node IDs
                - {source}_features: Feature names for each data source
                - {source}_node_coverage: Number of nodes with data for each source
                - {source}_feature_dim: Feature dimension for each source
        """
        metadata = {
            "num_nodes": len(self.global_node_ids),
            "num_timestamps": self.num_timestamps,
            "window_size": self.total_window_size,
            "stride": self.stride,
            "total_windows": len(self),
        }

        # Add information for each data source
        for source, info in self.dynamic_info.items():
            metadata[f"{source}_features_name"] = info["features_name"]
            metadata[f"{source}_node_coverage"] = np.sum(info["has_feature_mask"])
            metadata[f"{source}_feature_dim"] = len(info["features_name"])

        return metadata

class MadridFSDataset4NP(MadridDataset):
    """Dataset that only model the road segment that has fixed sensors as graph nodes. organized for spatio-temporal neural processing

    Args:
        MadridDataset (MadridDataset): The original dataset that contains all road segments [segments that have at least one dynamicsource of data] as graph nodes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Preserve the parent graph's node order when relabeling its edges.
        keep = torch.isin(self.full_graph.n_idx,
                          torch.from_numpy(self.active_node_global_idx))
        self.full_graph = self.full_graph.subgraph(keep)
        self.full_edge_index = self.full_graph.edge_index
        self.full_weights = self.full_graph.edge_attr
        self.active_node_global_idx = self.full_graph.n_idx.numpy()
        if self.split != "test":
            self.train_node_global_idx = self.full_graph.n_idx
