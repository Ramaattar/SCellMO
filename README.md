# SCellMO
Implementation code of SCellMO: Next cell mobility prediction and stolen phone detection

## Data Preparation

The model expects pre-processed `.npy` files (saved as NumPy dictionary objects) with the following keys:

| Key | Description |
|-----|-------------|
| `input_ids` | Cell-tower ID sequences |
| `target_id` | Next cell-tower ID target |
| `advertiser_tok` | User ID |
| `input_area` | Sequence of input tracking area |
| `target_area` | Next tracking area target |
| `input_hour` | Hour-of-day (0–23) |

You also need two vocabulary JSON files mapping tokens to indices (`vocab.json`, `area_vocab.json`).

Update the paths in `config.yaml` to point to your data files.

## Training

Training uses PyTorch Distributed Data Parallel (DDP) across multiple GPUs:

Key training features:
- **Adaptive area loss**: Area loss is toggled off when validation area loss increases
- **Warmup + ReduceLROnPlateau**: Linear LR warmup followed by plateau-based decay
- **Early stopping**: Training halts when LR drops below threshold
- **Checkpointing**: Saves best model by accuracy and by loss

Configure GPUs, batch size, learning rate, and other settings in `config.yaml`.

## Inference

Outputs (will be saved to `results/inference/`):
- `inference_metrics.json` — Top-k accuracy and MRR
- `inference_summary.csv` — Per-sample predictions
- `inference_results.pkl` — Full results with metrics

