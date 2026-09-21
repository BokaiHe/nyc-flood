"""Build the presentation notebook from the project's final experiment artifacts."""
from pathlib import Path
import re
import sys
import nbformat
from nbconvert import HTMLExporter
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'exports').mkdir(exist_ok=True)
cells = []


def md(text):
    cells.append(nbformat.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbformat.v4.new_code_cell(text.strip(), metadata={"jupyter": {"source_hidden": True}}))


md('''
<div class="summary-hero">
<div class="eyebrow">PROJECT SUMMARY · SENTINEL-2 / U-NET</div>
<h1>从卫星影像到水体分割</h1>
<p class="subtitle">RGB与RGB＋NIR对照实验 · 纽约市应用案例</p>
<p><a href="00_project_summary_en.ipynb">English summary</a></p>
<p>以公开标注数据完成模型训练与评价，再将固定模型应用到纽约市，观察增加近红外信息带来的变化。</p>
</div>

**核心问题：同样使用U-Net，增加一个近红外通道，能否改善水体分割？这种改善在NYC案例中如何体现？**

本报告按 **数据 → 方法 → 实验 → 定量结果 → NYC应用 → 结论** 展开。主结果统一采用最终08实验，历史调试过程保留在原Notebook中。
''')
code('''
from pathlib import Path
import csv, json, html
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Image, Markdown

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
OUT = ROOT / 'outputs/project_summary'
SOURCES = OUT / 'source_images'
RUN = ROOT / 'outputs/cloud_logs/nyc_inference/nyc_l1c_20260921_013008_456262'
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.loads((RUN / 'inference_config.json').read_text(encoding='utf-8'))
with (ROOT / 'outputs/nyc_point_validation/point_matches.csv').open(encoding='utf-8-sig') as f:
    points = list(csv.DictReader(f))
plt.rcParams.update({
    'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 1.2, 'legend.frameon': False, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'figure.dpi': 120,
})

import sys, base64
sys.path.insert(0, str(ROOT / 'scripts'))
from paper_figures import build_all
metrics = [
    dict(model='rgb', split='test', epoch=18, mean_iou=.2028, global_iou=.3698, f1=.5399, precision=.4374, recall=.7052),
    dict(model='rgb_nir', split='test', epoch=18, mean_iou=.5150, global_iou=.7247, f1=.8404, precision=.7507, recall=.9544),
    dict(model='rgb', split='bolivia', epoch=18, mean_iou=.3337, global_iou=.4594, f1=.6296, precision=.4761, recall=.9293),
    dict(model='rgb_nir', split='bolivia', epoch=18, mean_iou=.5793, global_iou=.7908, f1=.8832, precision=.8263, recall=.9484),
]

PAPER = build_all(metrics, points)
COLORS = {'rgb': '#3775BA', 'rgb_nir': '#B64342'}

def figure(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight')
    plt.show()
    plt.close(fig)

def table(rows, columns):
    header = ''.join(f'<th>{html.escape(title)}</th>' for key, title in columns)
    def fmt(v):
        return f'{v:.4f}' if isinstance(v, float) else str(v)
    body = ''.join('<tr>' + ''.join(f'<td>{html.escape(fmt(r[key]))}</td>' for key, _ in columns) + '</tr>' for r in rows)
    display(HTML('<div class="table-scroll"><table><thead><tr>' + header + '</tr></thead><tbody>' + body + '</tbody></table></div>'))

def caption(text):
    display(HTML(f'<p class="caption">{html.escape(text)}</p>'))

display(HTML('<style>.jp-RenderedHTMLCommon table{border-collapse:collapse;border-top:1.5px solid #222;border-bottom:1.5px solid #222;width:100%;}.jp-RenderedHTMLCommon th{background:white;border-bottom:1px solid #555;padding:8px;}.jp-RenderedHTMLCommon td{padding:7px 8px;border:none;}.jp-RenderedHTMLCommon tr{background:white;}.caption{font:12px Arial,sans-serif;color:#333;line-height:1.55;}details img{max-width:100%;height:auto;}</style>'))

''')

code('''
display(Image(filename=str(ROOT / 'outputs/method_figures/figure_1.png')))
caption('Overview | (a) RGB and RGB+NIR experiments on Sen1Floods11; image-mean water IoU averages the water-class IoU across images. (b) Fixed-model application to NYC, illustrated at Davenport. FloodNet counts summarize 13 wet site–date pairs across two dates; these are point detections, distinct from pixel metrics. Water masks include permanent water.')
''')

md('''
## 01 · 数据：用公开标签训练，用NYC场景观察应用效果

实验包含两类互补数据。**Sen1Floods11提供影像和人工像素标签，用于训练及定量评价；NYC提供同期影像和地面积水记录，用于应用分析。**

| 数据部分 | 使用范围 | 在研究中的作用 |
|---|---|---|
| Sen1Floods11人工标注子集 | 446组Sentinel-2影像与标签，原图512×512 | 建立水体分割模型并计算测试指标 |
| 输入通道 | RGB：B4/B3/B2；RGB＋NIR：B4/B3/B2/B8 | 构成三通道与四通道对照 |
| 参考标签 | 水体=1、非水体=0、忽略区域=−1 | 在共同有效像素上训练和评价 |
| NYC案例 | 2024-09-21与2024-10-18；13个站点—日期、10个站点 | 检查模型在城市沿海环境的输出 |
| FloodNet | 与影像过境时间匹配的积水深度序列 | 将模型输出与地面点观测联系起来 |

下面展示同一幅训练影像的RGB、NIR、人工标签和标签叠加。增加NIR并没有改变标签或研究对象，而是为同一分割任务增加一层光谱信息。
''')
code('''
display(Image(filename=str(PAPER['training'])))
caption('Figure 1 | India_285297 training example: RGB, NIR, human labels and the water-label overlay. RGB uses a 2–98% display stretch; NIR uses a fixed reflectance range of 0–0.6. Labels are unchanged.')
display(Image(filename=str(PAPER['splits'])))
caption('Figure 2 | Official data partition: 252 training, 89 validation, 90 test and 15 Bolivia images. Both experiments use the same partition.')
''')

md('''
## 02 · 方法：相同U-Net，比较三通道与四通道输入

U-Net先通过编码器提取由局部到整体的影像特征，再通过解码器恢复空间细节。跳跃连接把浅层的位置信息传回解码端，最终为每个像素输出水体与非水体两类分数。

采用WorldFloods公开实现中的U-Net结构：**64 → 128 → 256 → 512**通道，三次下采样，双线性上采样，卷积后使用ReLU。将公开方法适配到Sen1Floods11和本项目的3/4通道输入。
''')
code('''
display(Image(filename=str(ROOT / 'outputs/method_figures/figure_2.png')))
caption('Figure 3 | U-Net training and inference. RGB uses B4/B3/B2; RGB+NIR adds B8. Both use TRAIN-derived standardization. Weighted CE and Dice use valid labels. Training pair: India_285297. Inference inputs and RGB+NIR outputs: Davenport, 21 September 2024. The two models share the architecture and training protocol, with independently learned weights.')
''')
md('''
从输入到输出的处理是：

**影像DN → 反射率 → 训练集统计量标准化 → U-Net → 水体概率 → 水／非水掩膜。**

| 环节 | 选择 | 理由 |
|---|---|---|
| 通道组合 | B4/B3/B2 vs B4/B3/B2/B8 | 直接检验增加NIR后的表现 |
| 数值处理 | 训练数据DN/10000；本批NYC按产品元数据(DN−1000)/10000 | 将各自产品编码转换为反射率 |
| 标准化 | 使用训练集共同有效像素的逐通道均值、标准差 | 两组遵循同一预处理原则，应用时保持固定 |
| 有效像素 | 两组共用有效标签及四波段有效区域 | 保持评价像素一致 |
| 输出 | 每像素2类logits；softmax得到水体概率，argmax得到掩膜 | 将连续模型输出转换为分割结果 |
| NYC展示 | 512×512输入，导出中心128×128、10m网格 | 利用周边上下文，展示站点附近1.28km窗口 |

两组分别训练。除输入通道及相应输入层外，共享结构、划分、训练预算和选模规则。
''')

md('''
## 03 · 实验：采用公开实现设置，完成固定预算对照

每张512×512训练图划分为4个不重叠的256×256窗口，共1008个训练窗口。两组在RTX 4090上各训练25轮，以验证集Dice损失选择最佳权重；实际选中的轮次均为18。

| 参数 | 最终设置 | 选择理由 |
|---|---|---|
| 优化器与学习率 | Adam，1×10⁻⁴；weight decay=0 | 沿用所选公开实现配置 |
| Batch size / Epochs | 32 / 25 | 使用发布配置中的训练预算，两组一致 |
| 损失函数 | 0.5×加权交叉熵＋0.5×Dice | 同时优化像素分类与分割重叠 |
| 类别权重 | 训练有效像素N/n_c：非水约1.105、水约10.515 | 依据本数据的类别比例计算 |
| 学习率调度 | 验证Dice不改善时衰减；factor=0.5、patience=2 | 根据验证反馈调整步长 |
| 最佳权重 | 验证Dice损失最小 | 统一两组模型选择规则 |
| Seed / 增强 | 12 / 无额外增强 | 固定实现条件，使用同形状层的共同初始权重 |
| 执行方式 | 单卡、混合精度 | 在4090上完成实验 |

评价同时报告平均水体IoU和全局水体IoU：前者逐幅计算再平均，后者汇总全部有效像素后计算。F1综合精确率与召回率；Precision关注误报，Recall关注漏检。
''')
code('''
display(Image(filename=str(OUT.parent / 'paper_figures/03_training_record.png')))
caption('Figure 4 | Training and validation loss (a) and validation mean water IoU (b). Complete curve panels are reassembled from the original cloud record; curve pixels, axis values and colors are unchanged. Numerical epoch logs were not supplied.')
''')
md('''
从训练过程看，两组损失均随训练降低，RGB＋NIR的验证IoU维持在更高水平。下面使用同一实验选出的最佳权重，比较公开测试集的最终表现。
''')

md('''
## 04 · 公开测试结果：增加NIR后，分割指标提高

先看90幅正式测试图，再看15幅Bolivia专项测试图。所有数值来自最终08实验，下面的图和表采用统一指标口径。
''')
code('''
# Final notebook-08 cloud table, transcribed from the supplied original screenshot.
# The original is archived in source_images/metrics_table.png.
metrics = [
    dict(model='rgb', split='test', epoch=18, mean_iou=.2028, global_iou=.3698, f1=.5399, precision=.4374, recall=.7052),
    dict(model='rgb_nir', split='test', epoch=18, mean_iou=.5150, global_iou=.7247, f1=.8404, precision=.7507, recall=.9544),
    dict(model='rgb', split='bolivia', epoch=18, mean_iou=.3337, global_iou=.4594, f1=.6296, precision=.4761, recall=.9293),
    dict(model='rgb_nir', split='bolivia', epoch=18, mean_iou=.5793, global_iou=.7908, f1=.8832, precision=.8263, recall=.9484),
]
with (OUT / 'final08_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f,fieldnames=list(metrics[0]))
    writer.writeheader(); writer.writerows(metrics)
labels = {'rgb':'RGB', 'rgb_nir':'RGB+NIR'}
table([{**r, 'model': labels[r['model']], 'split': r['split'].title()} for r in metrics], [('split','Split'),('model','Input'),('mean_iou','Image-mean water IoU'),('global_iou','Global water IoU'),('f1','F1'),('precision','Precision'),('recall','Recall')])
display(Image(filename=str(PAPER['benchmark'])))
caption('Table 1 / Figure 5 | Final test metrics in a selectable table; both IoU columns refer to water. Panels (a–b) compare all five metrics for the same selected weights. Single run per input, seed 12; no between-run uncertainty is estimated. Values are transcribed from the original final-experiment table.')
''')
md('''
正式测试集上，RGB＋NIR的平均水体IoU从 **0.2028提高到0.5150**，F1从 **0.5399提高到0.8404**。Precision和Recall也同时提高，说明本次结果同时改善了误报与漏检表现。

Bolivia专项测试中也观察到同方向的提升。两种指标口径互为补充：较高的全局IoU说明总体像素重叠较好，而逐图平均IoU仍提示不同场景之间存在表现差异。
''')
code('''
display(Image(filename=str(OUT.parent / 'paper_figures/04b_test_prediction_record.png')))
caption('Figure 6 | Ghana_313799: RGB, NIR, labels and model predictions, reassembled from complete panels in the original cloud-rendered record. Only surrounding layout and headings have changed; no predicted pixels are reconstructed.')
''')
md('''
这幅示例中，RGB＋NIR对右下方水体的恢复更接近人工标签，细窄水道及局部边界仍有遗漏。定量指标说明整体差异，图像对照帮助定位差异发生在哪里。

接下来将这两组固定模型应用到NYC，观察面对城市、海湾与街道积水混合场景时的输出。
''')

md('''
## 05 · NYC应用：把固定模型带到城市沿海场景

选取2024年9月21日和10月18日的Sentinel-2 L1C影像，覆盖13个站点—日期窗口。模型输出水体概率和水／非水掩膜；FloodNet提供相同位置、相近时刻的地面积水测量。

NYC窗口包含海湾、河道、建筑和街道。**蓝色表示模型预测水体，红十字表示积水传感器位置。** 这让我们既能观察水体空间结构，也能检查观测点对应像素的输出。

下面选Davenport同一站点的两期结果，以固定地点展示日期间变化。
''')
code('''
display(Image(filename=str(PAPER['locator'])))
caption('Figure 7 | (a) Ten NYC sensor locations. (b) Davenport: 5.12 km input context and the central inference window. (c) RGB+NIR prediction on 21 September 2024. Scale bars follow the raster geotransform. Borough outlines: NYC Department of City Planning; contextual cartography only.')
display(Image(filename=str(PAPER['nyc'])))
caption('Figure 8 | Davenport on two observation dates. Shared columns compare RGB, NIR and the two water masks. Both RGB images use one pooled 2–98% stretch; NIR uses 0–0.6 reflectance. Prediction grids are unchanged at 10 m. Blue: water; beige: non-water; grey: unavailable. Dates are not labeled as pre-/post-flood.')
probability_image = base64.b64encode(PAPER['probability'].read_bytes()).decode()
display(HTML('<details><summary>Supplementary probability maps</summary><img alt="Davenport water probabilities" src="data:image/png;base64,' + probability_image + '"><p class="caption">Both dates and both models use the same probability scale, 0–1.</p></details>'))
''')
md('''
## 06 · 地面观测对照：从空间图像回到具体站点

将卫星目录的UTC过境时间与FloodNet深度序列对齐，取前后两条实测水深作为依据。13个站点像素均可用，过境前后测量均超过1cm，相邻包围测量间隔为62–188秒。

下图先对比两种模型在积水观测点的检出情况，再用热图展示全部13组概率。RGB＋NIR检出9组，RGB检出1组；两个日期的表现存在差异。
''')
code('''
display(Image(filename=str(PAPER['points'])))
caption('Figure 9 | Ground-observation comparison. (a) Detected / observed wet pairs; bar heights show counts, with the observed sample size in each label. (b) All 13 pairs on a shared probability scale; * marks the exported water classification. IDs 09-1–09-8 and 10-1–10-5 refer to September 21 and October 18, 2024. Site names and measured depths are listed below.')

detail_rows=[]
for date in dict.fromkeys(r['date'] for r in points):
    for j,r in enumerate((r for r in points if r['date']==date),1):
        values=[f"{date[5:7]}-{j}",r['date'],r['sensor'],
                f"{float(r['depth_before_cm']):.2f}",f"{float(r['depth_after_cm']):.2f}",
                f"{float(r['rgb_probability']):.4f}",f"{float(r['rgb_nir_probability']):.4f}"]
        detail_rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
headers=['ID','Date','Site','Depth before (cm)','Depth after (cm)','RGB P(water)','RGB+NIR P(water)']
display(HTML('<details><summary>All 13 observations: sites, depths and model probabilities</summary>'
             '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+
             '</tr></thead><tbody>'+''.join(detail_rows)+'</tbody></table></div></details>'))
''')
md('''
NYC案例中，RGB＋NIR在更多积水观测点对应像素上给出水体判别；同时，表现随日期和地点变化。以Davenport为例，9月同期水深约45cm、NIR模型判水；10月同期约25cm、两个模型均判非水。这一例子展示了城市街道积水应用中仍需面对的尺度与场景差异。

## 07 · 最终结论

**第一，完成了可复现的方法流程。** 从公开标签数据、U-Net训练到像素评价，再到NYC实际影像推理，构成完整的实验与应用链路。

**第二，NIR在本次对照实验中带来明确收益。** 相同协议下，公开测试集的平均水体IoU、全局IoU、F1、Precision和Recall均提高，Bolivia专项测试也呈同方向变化。

**第三，NYC案例呈现了应用效果及场景差异。** 固定模型能够输出城市沿海水体分布，RGB＋NIR在更多同期积水点对应像素上判水，也存在未检出的点位。

本项目的成果定位为 **“卫星水体分割的通道对照与NYC应用案例”**。公开测试提供定量分割评价，NYC提供影像展示和点观测对照。NYC输出包含原有海湾、河道；这13个有积水观测点用于探索性分析，尚不代表全市分割精度或新增淹没范围。

---

### 数据与方法来源

- **训练与标签：** [Sen1Floods11作者数据仓库](https://github.com/cloudtostreet/Sen1Floods11)。
- **模型方法：** [WorldFloods论文](https://doi.org/10.1038/s41598-021-86650-z)与[所用U-Net源码版本](https://github.com/spaceml-org/ml4floods/blob/f21129d1de0786eddb6b95118ccc47598fc3c40b/ml4floods/models/architectures/unets.py)；结构与发布配置适配到本项目数据与输入通道。
- **NYC地面观测：** [FloodNet方法与数据说明](https://www.floodnet.nyc/methodology)、[街道积水事件表](https://data.cityofnewyork.us/d/aq7i-eu5q)。
- **制图参考：** [Global flood extent segmentation in optical satellite images](https://pmc.ncbi.nlm.nih.gov/articles/PMC10661555/)中的紧凑影像组图、地图尺度展示与三线表；以本项目实际数据重新编排，未引入论文中的实验结果。
- **地图底图：** [NYC Department of City Planning borough boundaries](https://data.cityofnewyork.us/City-Government/Borough-Boundaries/gthc-hcne)，仅用于站点位置示意。
- **流程图设计：** 参考WorldFloods [2021论文](https://doi.org/10.1038/s41598-021-86650-z)与[2023论文图5](https://doi.org/10.1038/s41598-023-47595-7)的流程布局，以本项目实际方法、影像与预测重新绘制。可编辑PPT为 `exports/nyc_method_figures.pptx`。
- **项目结果：** 最终08云端曲线、指标及预测截图；11实际预测回传包；12逐点时空匹配结果。训练曲线保留原图，指标按截图转录；NYC点位数据读取实际CSV与配置。

### 实验记录入口

| 内容 | 对应Notebook |
|---|---|
| 最终模型训练与测试 | 08_train_public_unet_rgb_vs_rgb_nir |
| 逐图结果与错误分析 | 09_results_and_error_analysis |
| NYC输入与固定模型推理 | 11_nyc_l1c_inference |
| 全部NYC点位时空匹配 | 12_nyc_floodnet_point_validation |

*本Summary整合现有实验成果，不启动新的训练。原始调试记录与历史实验继续保留。*
''')

code("""
gallery = ''.join('<img alt="NYC site-date atlas" src="data:image/png;base64,' + base64.b64encode(PAPER[f'gallery{i}'].read_bytes()).decode() + '">' for i in range(1,4))
display(HTML('<details><summary>Supplementary atlas: all 13 NYC site–date windows</summary><p class="caption">Original record order. All 13 windows retained; each five-row page uses a common RGB display stretch, fixed NIR range and shared mask colors. Grey areas denote unavailable pixels.</p>' + gallery + '</details>'))
""")

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'}
nb.metadata.language_info = {'name':'python'}
p = ROOT / 'notebooks/00_project_summary.ipynb'
km = KernelManager(kernel_name='python3')
km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
NotebookClient(nb, km=km, timeout=120, resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.write(nb, p)
body, _ = HTMLExporter(exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True).from_notebook_node(nb)
body = re.sub(r'<script\b[^>]*>[\s\S]*?</script>', '', body, flags=re.IGNORECASE)
css = '''
<style>
:root { --jp-ui-font-family: Arial,'Microsoft YaHei',sans-serif; --jp-content-font-family: Arial,'Microsoft YaHei',sans-serif; }
body {background:white!important;color:#151515;}
main {max-width:1100px;margin:22px auto!important;padding:24px 36px!important;background:white;border-radius:0;box-shadow:none;}
.jp-Notebook,.jp-Cell,.jp-OutputArea-output {padding:0!important;}
.jp-InputPrompt,.jp-OutputPrompt {display:none!important;}
.jp-RenderedHTMLCommon {font-size:15px;line-height:1.8;color:#151515;}
.jp-RenderedHTMLCommon h2 {font-size:23px;color:#111;margin:34px 0 15px;border-bottom:1px solid #999;padding-bottom:8px;}
.jp-RenderedHTMLCommon h3 {font-size:18px;color:#222;}
.summary-hero {border-bottom:1.5px solid #222;padding:8px 0 18px;margin-bottom:22px;}
.summary-hero h1 {font-size:32px!important;line-height:1.3;color:#111;margin:12px 0!important;}
.eyebrow {font-size:11px;letter-spacing:1.4px;color:#555;}
.subtitle {font-size:19px;color:#333;}
.caption {color:#333;font-size:12px;line-height:1.55;margin:5px 0 22px;}
.table-scroll {overflow-x:auto;}
table {width:100%;margin:15px 0 22px!important;border-collapse:collapse!important;font-size:13px!important;border-top:1.5px solid #222!important;border-bottom:1.5px solid #222!important;}
th {background:white!important;color:#111!important;font-weight:600;border-bottom:1px solid #555!important;text-align:left!important;padding:8px!important;}
td {padding:7px 8px!important;border:none!important;text-align:left!important;}
tr,tr:nth-child(odd),tr:nth-child(even),tbody tr:hover {background:white!important;}
.jp-RenderedImage img,details img {max-width:100%;height:auto;display:block;margin:8px auto;}
details {margin:15px 0 25px;border-top:1px solid #AAA;padding-top:10px;}
summary {cursor:pointer;font-weight:600;font-size:14px;}
@media(max-width:760px){main{padding:14px!important;margin:0!important;}.summary-hero h1{font-size:26px!important;}}
@media print{main{max-width:none;padding:0!important;margin:0!important;}h2{break-after:avoid;}img,table{break-inside:avoid;}}
</style>
'''
body = body.replace('</head>', css + '</head>')
body = body.replace('<title>Notebook</title>', '<title>卫星水体分割 · 项目Summary</title>')
body = re.sub(r'href="(\d\d_[^"/]+\.ipynb)"', r'href="../notebooks/\1"', body)
(ROOT / 'exports/00_project_summary.html').write_text(body, encoding='utf-8')
print(p)
print(ROOT / 'exports/00_project_summary.html')
from summary_english import write_english_summary
print(write_english_summary(nb, ROOT, css))
print(ROOT / 'exports/00_project_summary_en.html')
