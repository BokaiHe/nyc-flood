# Reproduction guide

The repository includes executable experiment code and saved notebook outputs for direct GitHub viewing. Notebook 00, 01, 10 and 12 were refreshed locally; 05 replays returned epoch CSVs; 08–09 display labeled final-experiment records; 11 includes refreshed input previews and all 13 returned inference images. Historical 02–04 and 06–07 remain source-only. No training or checkpoint inference was rerun during this publication update.
The English HTML report can be opened locally after cloning, without Python. Its tables are selectable.
The repository includes no satellite dataset, model weights, Python environment or cloud run directory.

## Setup

Use Python 3.12. Create a virtual environment, install a CUDA-compatible PyTorch build on the training
host, then install the notebook dependencies:

```bash
python -m venv .venv
# Activate .venv using your platform's activation command.
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --user --name nyc-flood --display-name "NYC flood"
```

The completed cloud work used PyTorch 2.3.0. Requirements allow later compatible packages;
they are not a fully pinned reconstruction of the original CUDA environment.

## Main workflow

1. Download the hand-labeled data and official splits:
   `python scripts/download_sen1floods11.py --subset hand`.
2. Open `notebooks/01_data_preview.ipynb` to inspect the input data.
3. Run `08_train_public_unet_rgb_vs_rgb_nir.ipynb` on a CUDA host. It trains both models
   from scratch for 25 epochs and writes configuration, histories, best/last weights and metrics
   into a new `runs/worldfloods_unet_*` directory. Earlier experiments are not prerequisites.
4. Use that run in `09_results_and_error_analysis.ipynb` for evaluation and error galleries.
5. For NYC, run `download_nyc_candidates.py` before `download_nyc_l1c.py`: the L1C preparation
   reuses the candidate central grid and quality mask. The candidate configuration also retains
   historical inspection scenes. `11_nyc_l1c_inference.ipynb` uses the trained 08 checkpoints.
6. `12_nyc_floodnet_point_validation.ipynb` matches returned prediction rasters with FloodNet
   event profiles. Update its `RUN` path to your inference output. It expects the two
   `profiles_optical_YYYY-MM-DD.json` caches and event metadata under `data/nyc/observations`.
   The event acquisition workflow is in `scripts/screen_nyc_flood_events.py`; review the
   notebook's named inputs before executing. The derived matches are included in `results/`.

The exact historical checkpoints and final-08 numeric epoch logs are not included. Re-running
training creates a new experiment; it does not recover the original weights. The original curve
and example prediction are retained in `figure_sources/`, and the final benchmark table is
transcribed in `results/final08_metrics.csv`.

## Figures and reports

`scripts/paper_figures.py` implements the shared typography, benchmark plot and point comparison.
Benchmark and point figures can be regenerated from the two result CSVs alone:

```python
import csv, sys
sys.path.insert(0, 'scripts')
from paper_figures import benchmark, point_comparison
with open('results/final08_metrics.csv', encoding='utf-8-sig') as f:
    metrics = list(csv.DictReader(f))
for row in metrics:
    for key in ['mean_iou', 'global_iou', 'f1', 'precision', 'recall']:
        row[key] = float(row[key])
with open('results/point_matches.csv', encoding='utf-8-sig') as f:
    points = list(csv.DictReader(f))
benchmark(metrics)
point_comparison(points)
```

The full report builder also needs the original local imagery, NYC prediction rasters, map
boundaries and source figures at the paths stated in its code. It is not a data downloader.
`scripts/export_notebooks.py` exports available notebook outputs without training.

## Data and method attribution

- [Sen1Floods11 imagery, hand labels and official splits](https://github.com/cloudtostreet/Sen1Floods11).
- [WorldFloods paper](https://doi.org/10.1038/s41598-021-86650-z) and
  [referenced U-Net implementation](https://github.com/spaceml-org/ml4floods/blob/f21129d1de0786eddb6b95118ccc47598fc3c40b/ml4floods/models/architectures/unets.py).
- [FloodNet methodology](https://www.floodnet.nyc/methodology) and
  [NYC street-flood events](https://data.cityofnewyork.us/d/aq7i-eu5q).
- [NYC borough boundaries](https://data.cityofnewyork.us/City-Government/Borough-Boundaries/gthc-hcne).

Figures use project data and model outputs. Dataset and third-party implementation terms remain
with their respective sources; publication of this repository does not relicense them.

Notebook display images are limited to 1280 pixels in width; larger panels use JPEG display copies to keep GitHub previews lightweight. Small charts retain PNG. Original local figures and analysis rasters are unchanged.
