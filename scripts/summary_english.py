"""English Summary, sharing the executed Chinese Summary's code and evidence."""
from copy import deepcopy
import re
import nbformat
from nbconvert import HTMLExporter

ENGLISH_MARKDOWN = {
0: """<div class="summary-hero">
<div class="eyebrow">PROJECT SUMMARY · SENTINEL-2 / U-NET</div>
<h1>Satellite Water Segmentation</h1>
<p class="subtitle">RGB versus RGB+NIR · A New York City case study</p>
<p>Train and evaluate on public pixel labels, then apply the fixed models to NYC imagery and compare their outputs with ground observations.</p>
</div>

**Research question: How does adding near-infrared information change U-Net water segmentation, and how does this difference appear in an NYC application?**

The report follows **data → method → experiment → benchmark results → NYC application → conclusions**. All main model results use the final experiment in Notebook 08.
""",
3: """## 01 · Data: labeled imagery and an NYC application

Two complementary sources support the study. **Sen1Floods11 supplies imagery and pixel labels for supervised training and evaluation. NYC imagery and FloodNet measurements support the application analysis.**

| Component | Coverage | Role |
|---|---|---|
| Sen1Floods11 hand-labeled subset | 446 Sentinel-2 image/label pairs, 512 × 512 pixels | Training and quantitative segmentation evaluation |
| Input bands | RGB: B4/B3/B2; RGB+NIR: B4/B3/B2/B8 | A three-band versus four-band comparison |
| Pixel labels | Water = 1, non-water = 0, ignored = -1 | Supervision and evaluation on shared valid pixels |
| NYC cases | 13 site–date windows at 10 sites; 21 September and 18 October 2024 | Application to coastal urban scenes |
| FloodNet | Depth sequences aligned with satellite overpass times | Ground observations at the sensor locations |

The training example below shows RGB, NIR, the human label and a label overlay. Adding NIR changes the available spectral information while retaining the same task and target labels.
""",
5: """## 02 · Method: the same U-Net with three or four bands

U-Net combines an encoder that extracts features at progressively coarser scales with a decoder that restores spatial detail. Skip connections carry fine-scale features into the decoder. The output is a non-water/water score pair for every pixel.

We adapt the public WorldFloods U-Net implementation: **64 → 128 → 256 → 512** channels, three downsampling stages, bilinear upsampling and ReLU activations. The two models use three or four input bands and learn their weights independently.
""",
7: """The processing sequence is:

**Image DN → reflectance → training-set standardization → U-Net → water probability → water/non-water mask.**

| Component | Implementation | Purpose |
|---|---|---|
| Band comparison | B4/B3/B2 versus B4/B3/B2/B8 | Measure the effect of adding NIR |
| Reflectance conversion | Training: DN/10000; these NYC products: (DN−1000)/10000 | Respect each product's encoding |
| Standardization | Per-band training mean and standard deviation | Keep preprocessing fixed during evaluation and application |
| Valid pixels | Shared label and four-band validity masks | Evaluate both models on the same pixels |
| Output | Two logits per pixel; softmax probabilities and argmax masks | Convert model scores into segmentation outputs |
| NYC windows | 512 × 512 context; central 128 × 128 export at 10 m | Map a 1.28 km neighborhood with surrounding context |

The models share the architecture, split, training budget and selection rule, apart from their input channels and corresponding first-layer dimensions.
""",
8: """## 03 · Experiment: a fixed public-method protocol

Each 512 × 512 training image supplies four non-overlapping 256 × 256 windows, giving **1008 training windows**. Both models trained for **25 epochs on an RTX 4090**. Minimum validation Dice loss selected epoch **18** for each model.

| Setting | Final value | Basis |
|---|---|---|
| Optimizer | Adam, learning rate 1×10⁻⁴, weight decay 0 | Selected public implementation |
| Batch size / epochs | 32 / 25 | Published training budget, shared by both models |
| Objective | 0.5 weighted cross-entropy + 0.5 Dice | Pixel classification and overlap |
| Class weights | Training N/n_c; approximately 1.105 non-water and 10.515 water | Class counts in the current dataset |
| Scheduler | Validation Dice plateau; factor 0.5, patience 2 | Validation-driven learning-rate adjustment |
| Model selection | Minimum validation Dice loss | The same rule for both inputs |
| Seed / augmentation | 12 / no extra augmentation | Fixed conditions and shared matching-layer initialization |
| Runtime | Single GPU, mixed precision | Execution on the available hardware |

**Image-mean water IoU** averages image-level overlap; **global water IoU** pools valid pixels before computing overlap. Precision reflects false positives, recall reflects missed water, and F1 combines both. The label mask is applied to both probabilities and targets in Dice, a documented adaptation of the public loss.
""",
10: """Both losses decrease during training, while RGB+NIR maintains higher validation IoU. The next section compares the validation-selected checkpoints on the same public test splits.
""",
11: """## 04 · Benchmark results: the contribution of NIR

Evaluate the **90-image test split** and the **15-image Bolivia split** separately. All values below come from the final Notebook 08 experiment and use the same metric definitions.
""",
13: """On the test split, adding NIR increases image-mean water IoU from **0.2028 to 0.5150** and F1 from **0.5399 to 0.8404**. Precision and recall also increase, indicating improvements in both false-positive and missed-water behavior in this experiment.

The Bolivia split shows the same direction of improvement. Global and mean IoU describe complementary aspects of performance: pooled pixel overlap and variation across individual images.
""",
15: """In this example, RGB+NIR recovers the lower-right water body more closely to the human label, while narrow channels and some boundaries remain incomplete. The benchmark metrics quantify the difference; the image comparison shows where it occurs.

We next apply the two fixed models to NYC's urban coastal scenes.
""",
16: """## 05 · NYC application: coastal urban water mapping

Sentinel-2 L1C imagery from **21 September and 18 October 2024** covers 13 site–date windows. The models produce water probabilities and masks, while FloodNet records ground-level water depth at the corresponding sensor locations and times.

The windows contain bays, channels, buildings and streets. **Blue denotes predicted water; red crosses locate the FloodNet sensors.** This supports both a spatial view of water predictions and a comparison at observed locations.

The following panels show Davenport on both dates, holding the location fixed to illustrate temporal variation.
""",
18: """## 06 · Ground observations: comparison at sensor locations

Align catalog overpass times in UTC with FloodNet depth sequences and retain the measurements immediately before and after each overpass. All 13 sensor pixels are usable, and both bracketing measurements exceed 1 cm; measurement intervals range from **62 to 188 seconds**.

The bars summarize wet-site detections; the heatmap retains all 13 probability pairs. **RGB+NIR detects 9 pairs, compared with 1 for RGB**, with different results across the two dates.
""",
20: """RGB+NIR predicts water at more of the observed wet-site pixels, although performance varies by date and location. At Davenport, the September measurements are approximately 45 cm and RGB+NIR predicts water; in October, the measurements are approximately 25 cm and neither model predicts water. This illustrates the scene and scale challenges of applying satellite water segmentation to street flooding.

## 07 · Conclusions

**A complete experimental workflow.** The project connects labeled satellite data, U-Net training, pixel-level evaluation, NYC inference and ground-observation matching.

**NIR improves the final benchmark comparison.** Under the shared protocol, RGB+NIR improves mean/global water IoU, F1, precision and recall on both the test and Bolivia splits.

**NYC demonstrates application behavior and variation.** The fixed models map coastal water and give different responses at co-temporal wet-site observations. RGB+NIR detects more of the selected pairs, with remaining disagreements.

The study is framed as **a spectral-input comparison for satellite water segmentation, with an NYC application case study**. Public labels provide quantitative segmentation evaluation; NYC provides spatial examples and a selected wet-site comparison. NYC predictions include permanent bays and channels, so these observations do not establish citywide segmentation accuracy or newly inundated area.

### Data and method sources

- **Training imagery and labels:** [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11).
- **Model:** [WorldFloods](https://doi.org/10.1038/s41598-021-86650-z) and the [pinned U-Net implementation](https://github.com/spaceml-org/ml4floods/blob/f21129d1de0786eddb6b95118ccc47598fc3c40b/ml4floods/models/architectures/unets.py), adapted to this dataset and input comparison.
- **Ground observations:** [FloodNet methodology](https://www.floodnet.nyc/methodology) and the [NYC street-flood event data](https://data.cityofnewyork.us/d/aq7i-eu5q).
- **Figure design:** [Global flood extent segmentation in optical satellite images](https://pmc.ncbi.nlm.nih.gov/articles/PMC10661555/) informed the compact image plates, map context and three-rule tables. Figures use this project's data and results.
- **Map context:** [NYC Department of City Planning borough boundaries](https://data.cityofnewyork.us/City-Government/Borough-Boundaries/gthc-hcne).
- **Workflow diagrams:** original project diagrams informed by the WorldFloods [2021](https://doi.org/10.1038/s41598-021-86650-z) and [2023](https://doi.org/10.1038/s41598-023-47595-7) layouts. Editable source: `exports/nyc_method_figures_en.pptx`.
- **Experimental evidence:** original Notebook 08 cloud screenshots, returned Notebook 11 prediction files and Notebook 12 matched observations. Benchmark numbers were transcribed from the original table; NYC values are read from the returned CSVs and rasters.

### Reproducible notebook workflow

| Stage | Notebook |
|---|---|
| Data overview | [01 · Data preview](01_data_preview.ipynb) |
| Final training and testing | [08 · Public U-Net](08_train_public_unet_rgb_vs_rgb_nir.ipynb) |
| Image-level results and errors | [09 · Error analysis](09_results_and_error_analysis.ipynb) |
| NYC inputs and fixed-model inference | [11 · NYC inference](11_nyc_l1c_inference.ipynb) |
| Spatial and temporal observation matching | [12 · FloodNet comparison](12_nyc_floodnet_point_validation.ipynb) |

Notebooks 02–07 document the earlier experiments and diagnostics; Notebook 10 records the initial input audit. This Summary compiles existing results without training a model.
""",
}


def write_english_summary(nb, root, css):
    english = deepcopy(nb)
    markdown_indices = {i for i,c in enumerate(english.cells) if c.cell_type == 'markdown'}
    if markdown_indices != set(ENGLISH_MARKDOWN):
        raise ValueError('Summary sections changed: update the English narrative mapping.')
    for i, source in ENGLISH_MARKDOWN.items():
        english.cells[i].source = source.strip()
    english.metadata['title'] = 'Satellite Water Segmentation: Project Summary'
    english.metadata['presentation_language'] = 'en'
    path = root / 'notebooks/00_project_summary_en.ipynb'
    nbformat.write(english, path)
    body,_ = HTMLExporter(exclude_input=True, exclude_input_prompt=True,
                         exclude_output_prompt=True).from_notebook_node(english)
    body = re.sub(r'<script\b[^>]*>[\s\S]*?</script>', '', body, flags=re.IGNORECASE)
    body = body.replace('</head>',css+'</head>')
    body = body.replace('<title>Notebook</title>','<title>Satellite Water Segmentation · Project Summary</title>')
    # Notebook-relative links need a sibling directory in the standalone HTML.
    body = re.sub(r'href="(\d\d_[^"/]+\.ipynb)"',r'href="../notebooks/\1"',body)
    (root/'exports/00_project_summary_en.html').write_text(body,encoding='utf-8')
    return path
