# MMTD data loaders

This is a standalone adaptation of the dataset, spatio-temporal sampler, and
PyG loader used in [TA-ANP](https://github.com/qishenzhou/TA-ANP). It loads
**prepared HDF5 files**. Fixed-sensor loading works without TomTom data.

## Run the example

Use Python 3.11 and install PyTorch for your CPU/CUDA environment, then run from
the repository root:

```bash
python -m pip install -r requirements.txt
python -m examples.load_fixed_sensor --data-root /path/to/MMTD
```

The data directory must contain `distance_matrix.h5`, `road_info.h5`, and
`fixed_sensor_data.h5` from the [Zenodo release](https://doi.org/10.5281/zenodo.18781181).
Read each file's embedded `README` for its metadata. Input files are opened
read-only. No clustering CSV or generated cache files are required.

The example uses a 12-step history, a 4-step prediction horizon, a 1 km edge
threshold, and two temporal windows per batch. Use `--split train`, `--split val`,
or `--split test` (default), and `--no-scale` to retain original units.

## Interfaces and inputs

- `MadridDataset`: all road segments in the distance matrix form the graph.
- `MadridFSDataset4NP`: restricts that graph to roads with fixed sensors; this is
  the default in the example.
- `NPSampler1`: combines temporal windows with spatial subgraphs and samples
  observed/unobserved nodes during training and validation.
- `STDataLoader`: collates those samples into PyG batches and exposes target
  normalization/inverse normalization.

The loader retains the TA-ANP prepared-data layout:

| File | Required datasets |
| --- | --- |
| `distance_matrix.h5` | `data/values` as an N × N distance matrix in km; `metadata/nodes` |
| `road_info.h5` | `data/values` as N × F static attributes; `metadata/nodes`; `metadata/features` |
| `fixed_sensor_data.h5` | `data/values` as T × Ns × 3; `metadata/nodes`; `metadata/features`; `metadata/times` |

Road attributes must follow the distance matrix's node order and the original
MMTD feature order: the first 12 columns are static road attributes, columns
12–13 are coordinates, and columns 15, 17, …, 33 are POI densities. The named
continuous attributes used for scaling are listed in `MadridDataset._scale_data`.
Sensor columns are flow, occupancy, and congestion level; timestamps are strings
in `YYYY-MM-DD HH:MM:SS` format. `target_features="q"`, `"o"`, or `"c"` selects
the target column. `input_features` is retained for API compatibility; `x`
contains all three sensor columns.

## Sampling, masks, and outputs

The default chronological split is 60% training, 20% validation, and 20% test.
Windows stay within their split. `permanent_unobs_ratio` and `seed4unob` control
held-out sensor nodes: they are removed from training/validation graphs and
retained as unobserved targets at test time. The sampler additionally chooses
random context/target subsets during training and validation. Seed Python's
`random`, NumPy, and PyTorch if you need reproducible training batches.

Without a CSV, `num_cluster` divides graph nodes into deterministic contiguous
groups. These are a simple batching fallback, not geographic clusters. To reuse
your own spatial clustering, pass `cluster_path` to `NPSampler1` or
`--cluster-path /path/to/clusters.csv` to the example. The CSV needs `node_id`
and `cluster_id_road_graph`, with one assignment for every dataset graph node.
Each spatial batch contains one group; `s_batch_size` is retained for compatibility.
Cross-group edges are omitted. Singleton groups have no additional random holdout.

For a single sample with N graph nodes and W = history + prediction steps:

| Field | Meaning |
| --- | --- |
| `x` | N × W × 3 sensor inputs; held-out nodes and future steps are NaN |
| `y` | Ns × W × 1 targets for sensor-covered nodes, including historical steps |
| `graph_mask` | N booleans selecting nodes represented in `y` |
| `label_unobs_mask` | N booleans indicating withheld nodes |
| `edge_index`, `edge_attr` | Local graph connectivity and distance-derived weights |
| `n_id` | Original road-segment IDs |
| `static_aux`, `poi_aux`, `pos`, `time_enc` | Road, POI, coordinate, and calendar features |

PyG concatenates node dimensions across temporal samples. Use `batch`/`ptr` to
identify each graph. To select withheld labels on a full-network batch, use
`label_unobs_mask[graph_mask]`. Missing measurements remain NaN; combine masks
with `torch.isfinite(y)` when computing losses. Forecast targets are the last
`pred_window` steps of `y` (the example uses four).

This adaptation retains the experiment's sensor flow filter: rows with flow
below 10 or above 300 are treated as missing. Dynamic scalers are fitted on the
training period; permanently held-out sensor nodes are excluded. Static scalers
use all road attributes. `not_scale=True` skips normalization, but retains the
flow filter and missing-data masks. Compatibility arguments `train_unobs_ratio`
and `stack_time` do not change sampler masking or the N × W × F layout.

## Optional, already aligned TomTom inputs

You may add `"tomtom": "/path/to/tomtom_data.h5"` to
`dynamic_features_paths`. The file must already use matching road IDs and the
same time rows as the sensor file, with the original feature order: average
travel time, median travel time, hits, speed standard deviation, then speed
statistics. This loader does not perform TomTom binding or map matching.

With this input, samples also include `dynamic_aux` (TomTom columns from hits
onward, history only) and `tom_hits` (hits over the whole target window).
Future `tom_hits` are auxiliary targets, not forecasting inputs. Roads lacking
TomTom coverage are padded with NaNs. Without TomTom data, both fields are absent.
TomTom transform helpers require TomTom inputs. Supply licensed data separately.

## Differences from the experiment implementation

The dataset interface and sample fields come from TA-ANP. This public adaptation
makes TomTom optional, replaces the hardcoded clustering path, builds graph
edges in row chunks, and computes scalers without experiment-specific caches
or plotting dependencies. It also handles missing TomTom coverage and applies
the same cleaning/speed normalization when fitting dynamic scalers. For exact
TA-ANP checkpoint reproduction, use the original project's loader and setup.

Dynamic data for the selected split are held in memory; fitting scalers also
reads the training period. The distance matrix is read in chunks, but retained
edges still occupy memory. Choose a suitable distance threshold and batch size.

## Validation and attribution

```bash
python -m unittest discover -s tests -v
```

Tests generate small HDF5 fixtures covering fixed-sensor loading, scaling,
held-out masks, optional TomTom coverage, CSV clusters, and empty windows.

`time_features_tool.py` retains its GluonTS/Amazon attribution and Apache-2.0
license header. The license text is in
[`licenses/Apache-2.0.txt`](../licenses/Apache-2.0.txt). Other loader code is
covered by the repository's MIT license.
