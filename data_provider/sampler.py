import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import Sampler
from .madrid_dataset import MadridDataset


class NPSampler1(Sampler):
    """
    Sampler for neural process training.
    This sampler is based on  "Le, T.A., Kim, H., Garnelo, M., n.d. Empirical Evaluation of Neural Process Objectives."
    the ratio of context nodes and target nodes m ~[0.1,0.9], n ~ [m+1, 1]

    Generates batch indices for both temporal and spatial dimensions,
    specifically designed for neural process.
    Each batch contains a combination of temporal indices and spatial indices.
    """

    def __init__(
        self,
        dataset: MadridDataset,
        t_batch_size: int,
        s_batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_cluster: int = 10,
        cluster_path: str | None = None,
    ):
        """_summary_

        Args:
            dataset (MadridDataset): pytorch dataset instance
            t_batch_size (int): batch size for temporal dimension
            s_batch_size (int): batch size for spatial dimension
            num_neighbors (tuple[int]): number of neighbors for each hop of the graph
            shuffle (bool, optional): whether to shuffle temporal indices. Defaults to True.
            drop_last (bool, optional): whether to drop the last temporal batch. Defaults to False.
            num_cluster (int, optional): number of clusters for the graph. Defaults to 10.
        """

        if t_batch_size < 1 or num_cluster < 1:
            raise ValueError("t_batch_size and num_cluster must be positive")
        self.dataset = dataset
        self.t_batch_size = t_batch_size
        self.s_batch_size = s_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_cluster = num_cluster
        # Get dataset length (temporal dimension)
        self.num_time_steps = dataset.len()

        # Calculate number of temporal batches
        if self.drop_last:
            self.num_t_batches = self.num_time_steps // self.t_batch_size
        else:
            self.num_t_batches = (self.num_time_steps + self.t_batch_size -
                                  1) // self.t_batch_size

        graph = self.dataset.full_graph
        cluster_data = []
        if cluster_path is not None:
            table = pd.read_csv(cluster_path)
            if not {"node_id", "cluster_id_road_graph"}.issubset(table.columns):
                raise ValueError("Cluster CSV needs node_id and cluster_id_road_graph columns")
            if table.node_id.duplicated().any() or table.cluster_id_road_graph.isna().any():
                raise ValueError("Each node must have exactly one cluster assignment")
            if not np.isin(graph.n_id.numpy(), table.node_id.to_numpy()).all():
                raise ValueError("Cluster CSV must cover every dataset graph node")
            for _, group in table.groupby("cluster_id_road_graph"):
                mask = torch.isin(graph.n_id, torch.tensor(group.node_id.to_numpy()))
                if mask.any():
                    cluster_data.append(graph.subgraph(mask))
        else:
            # A reproducible fallback without external clustering artifacts.
            for indices in torch.tensor_split(torch.arange(graph.num_nodes), num_cluster):
                if indices.numel():
                    cluster_data.append(graph.subgraph(indices))

        self.spatial_loader = cluster_data
        self.num_s_batches = len(self.spatial_loader)

    def __iter__(self):
        """
        Generate iterator for spatio-temporal batch indices.

        Returns:
            Iterator: Each iteration returns a batch containing spatio-temporal indices
        """
        # Generate temporal batches
        t_indices = list(range(self.num_time_steps))
        if self.shuffle:
            # Shuffle temporal indices if required
            random.shuffle(t_indices)

        # Group temporal indices into batches
        t_batches = [
            t_indices[i:i + self.t_batch_size]
            for i in range(0, len(t_indices), self.t_batch_size)
        ]

        # Drop last incomplete batch if required
        if self.drop_last and t_batches and len(t_batches[-1]) < self.t_batch_size:
            t_batches = t_batches[:-1]

        # For each combination of spatial and temporal batches
        for t_batch in t_batches:
            for _, s_batch in enumerate(self.spatial_loader):
                # Each batch is a list of indices, where each index is a tuple:
                # (temporal_index, current_spatial_batch, observed_nodes)
                if self.dataset.split != "test" and s_batch.num_nodes >= 2:
                    dynamic_context_num = max(
                        1, int(np.random.uniform(0.1, 0.9) * s_batch.num_nodes))
                    dynamic_subgraph_num = int(
                        np.random.uniform(dynamic_context_num + 1, s_batch.num_nodes))
                    dynamic_unobs_num = dynamic_subgraph_num - dynamic_context_num
                    sub_graph_node_index = torch.randperm(
                        s_batch.num_nodes)[:dynamic_subgraph_num]
                    # get sub_s_batch based on sub_graph_node_index
                    s_batch = s_batch.subgraph(sub_graph_node_index)
                    s_batch.unobs_global_idx = torch.from_numpy(
                        np.random.choice(
                            s_batch.n_idx,
                            size=dynamic_unobs_num,
                            replace=False,
                        ))  # control each cluster have same unobs rate
                else:
                    s_batch.unobs_global_idx = s_batch.n_idx[torch.isin(
                        s_batch.n_idx,
                        torch.from_numpy(self.dataset.permanent_unobs_nodes_global_idx))]
                yield [(t_idx, s_batch) for t_idx in t_batch]

    def __len__(self):
        """
        Calculate total number of batches.

        Returns:
            int: Total number of batches = number of temporal batches × number of spatial batches
        """

        # Total batches = temporal batches × spatial batches
        return self.num_t_batches * self.num_s_batches
