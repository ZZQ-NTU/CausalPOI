# CausalPOI

This folder splits the original single script into several files.

## Files

- `main.py`: entry point.
- `data_utils.py`: JSON loading and check-in dictionary construction.
- `dataset.py`: dataset and collate function.
- `graph_builder.py`: POI embedding loading/computation and graph construction.
- `models.py`: GATv2 encoder, GRU temporal encoder, and DragonNet prediction heads.
- `train_eval.py`: training, validation, and loss.
- `utils.py`: metrics, logging helpers, and positional encoding.

## Important changes

1. `function.json` is loaded and used as `alpha(p,n)` with key format `"pid,nid"`.
2. Treatment graph edge weight is `alpha(p,n) * spatial_decay`.
3. Control graph edge weight is `spatial_decay`.
4. Neighbor representation is computed by mean pooling all non-center nodes in each weekly graph.
5. Position encoding removes the extra outside multiplication by `lambda`.

## Run

Place these files in the same directory as `config.py`, then run:

```bash
python main.py -c Northeast --runs 10
```

The graph cache is saved as:

```text
dataset/{dataset}/pregraphs_functional.pkl
dataset/{dataset}/pregraphs_functional_parts/
```

This avoids accidentally loading old `pregraphs.pkl` files that do not contain `alpha(p,n)`.
