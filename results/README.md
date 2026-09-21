# Result definitions and provenance

`final08_metrics.csv` contains four model/split rows transcribed from the completed final
Notebook 08 result table. Both selected checkpoints are from epoch 18. The experiment used
one seed (12); no run-to-run uncertainty is estimated.

- `mean_iou`: **image-mean water IoU**, the mean of per-image water-class IoUs under the
  notebook's valid-pixel and empty-union handling. It is not an average over the two classes.
- `global_iou`: water IoU after pooling valid-pixel confusion counts across the split.
- `f1`, `precision`, `recall`: water-class metrics computed from those pooled counts.

`point_matches.csv` contains all 13 NYC site–date pairs at 10 sites, with timestamps,
bracketing observed depths, interpolated depths, quality fractions and both model outputs.
Depths are in centimetres; times are UTC. Model classifications are the exported fixed-model
decisions, not decisions retuned on these observations.

| Date | Observed wet pairs | RGB detected | RGB+NIR detected |
|---|---:|---:|---:|
| 2024-09-21 | 8 | 0 | 7 |
| 2024-10-18 | 5 | 1 | 2 |
| Total | 13 | 1 | 9 |

These are selected wet-point detection counts. They do not estimate NYC precision, pixel IoU,
or new flood extent. The masks include permanent water. Negative observations and independent
NYC pixel labels would support a broader validation.

Display revisions change labels, layout and count scaling only; they do not change the
metrics, probabilities, predictions, sample inclusion or selected model weights. Original
training curves retain their screenshot typography because numeric final-08 epoch logs
are not available locally. Web images are resized display derivatives, not analysis inputs.
