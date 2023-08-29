import os

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

font_set = FontProperties(fname=r"simsun.ttc", size=6)
font_set_1 = FontProperties(fname=r"simsun.ttc", size=10)

myfont = matplotlib.font_manager.FontProperties(
    fname=r'simsun.ttc')

save_dir = "main_curves"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

path = "Results.xlsx"  # this is the excel file containing the results (like the one we released)
file = pd.read_excel(path, sheet_name="imcls_fewshot")

config = {
    "font.family": 'myfont',
    "font.size": 18,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

datasets = [
    "OxfordPets", "Flowers102", "FGVCAircraft", "DTD",
    "EuroSAT", "StanfordCars", "Food101", "SUN397",
    "Caltech101", "UCF101", "ImageNet"
]

# datasets = [
#     "OxfordPets", "Flowers102", "FGVCAircraft", "DTD",
#     "EuroSAT", "StanfordCars", "Food101", "SUN397",
#     "Caltech101", "UCF101"
# ]

shots = [1, 2, 4, 8, 16]

COLORS = {
    "zs": "00",
    "linear": "C4",
    "CoOp_v16_end": "C0",
    "CoOp_v16_mid": "C2",
    "PTVE": "C1"
}
MS = 3
ALPHA = 1
plt.rcParams.update({"font.size": 12})

average = {
    "zs": 0.,
    "CoOp_v16_end": np.array([0., 0., 0., 0., 0.]),
    "CoOp_v16_mid": np.array([0., 0., 0., 0., 0.]),
    "PTVE": np.array([0., 0., 0., 0., 0.]),
    "linear": np.array([0., 0., 0., 0., 0.])
}

for dataset in datasets:
    print(f"Processing {dataset} ...")

    zs = file[dataset][0]

    CoOp_v16_end = file[dataset][2:7]
    CoOp_v16_end = [float(num) for num in CoOp_v16_end]

    CoOp_v16_mid = file[dataset][7:12]
    CoOp_v16_mid = [float(num) for num in CoOp_v16_mid]

    PTVE = file[dataset][12:17]
    PTVE = [float(num) for num in PTVE]

    linear = file[dataset][17:22]
    linear = [float(num) for num in linear]

    average["zs"] += zs
    average["CoOp_v16_end"] += np.array(CoOp_v16_end)
    average["CoOp_v16_mid"] += np.array(CoOp_v16_mid)
    average["PTVE"] += np.array(PTVE)
    average["linear"] += np.array(linear)

    # Plot
    values = [zs]
    values += linear
    values += CoOp_v16_end
    values += CoOp_v16_mid
    values += PTVE
    val_min, val_max = min(values), max(values)
    diff = val_max - val_min
    val_bot = val_min - diff * 0.05
    val_top = val_max + diff * 0.05

    fig, ax = plt.subplots()
    ax.set_facecolor("#FFFFFF")

    ax.set_xticks([0] + shots)
    ax.set_xticklabels([0] + shots)
    ax.set_xlabel('每个类别的训练样本数量', fontsize=14, fontfamily='sans-serif', fontstyle='italic',
                  fontproperties=font_set)
    ax.set_ylabel('分数（%）', fontsize=14, fontfamily='sans-serif', fontstyle='italic', fontproperties=font_set)
    ax.grid(axis="x", color="#EBEBEB", linewidth=1)
    ax.axhline(zs, color="#EBEBEB", linewidth=1)
    ax.set_title(dataset, fontsize=14, fontfamily='sans-serif', fontstyle='italic',
                 fontproperties=font_set)
    ax.set_ylim(val_bot, val_top)

    ax.plot(
        0, zs,
        marker="*",
        markersize=MS * 1.5,
        color=COLORS["zs"],
        alpha=ALPHA
    )
    ax.plot(
        shots, CoOp_v16_end,
        marker="o",
        markersize=MS,
        color=COLORS["CoOp_v16_end"],
        label="CLIP + CoOp (M=16, end)",
        linestyle="--",
        alpha=ALPHA
    )
    ax.plot(
        shots, CoOp_v16_mid,
        marker="o",
        markersize=MS,
        color=COLORS["CoOp_v16_mid"],
        label="CLIP + CoOp (M=16, mid)",
        linestyle="-.",
        alpha=ALPHA
    )
    ax.plot(
        shots, PTVE,
        marker="o",
        markersize=MS,
        color=COLORS["PTVE"],
        label="PTVE",
        alpha=ALPHA
    )
    ax.plot(
        shots, linear,
        marker="o",
        markersize=MS,
        color=COLORS["linear"],
        label="Linear probe CLIP",
        linestyle="dotted",
        alpha=ALPHA
    )

    # ax.text(-0.5, zs - diff * 0.11, "Zero-shot\nCLIP", color=COLORS["zs"])
    ax.text(-0.5, zs - diff * 0.05, "CLIP零样本评估", color=COLORS["zs"], fontproperties=font_set_1)
    ax.legend(loc="lower right")

    # fig.savefig(f"{save_dir}/{dataset}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_dir}/{dataset}.png", bbox_inches="tight")

# Plot
average = {k: v / len(datasets) for k, v in average.items()}
zs = average["zs"]
linear = list(average["linear"])
CoOp_v16_end = list(average["CoOp_v16_end"])
CoOp_v16_mid = list(average["CoOp_v16_mid"])
PTVE = list(average["PTVE"])

values = [zs]
values += linear
values += CoOp_v16_end
values += CoOp_v16_mid
values += PTVE
val_min, val_max = min(values), max(values)
diff = val_max - val_min
val_bot = val_min - diff * 0.05
val_top = val_max + diff * 0.05

fig, ax = plt.subplots()
# ax.rcParams['font.sans-serif'] = ['serif']
ax.set_facecolor("#FFFFFF")

ax.set_xticks([0] + shots)
ax.set_xticklabels([0] + shots)
# ax.set_xlabel("Number of labeled training examples per class")
ax.set_xlabel('每个类别的训练样本数量', fontsize=14, fontfamily='sans-serif', fontstyle='italic',
              fontproperties=font_set)
ax.set_ylabel('分数（%）', fontsize=14, fontfamily='sans-serif', fontstyle='italic', fontproperties=font_set)
ax.grid(axis="x", color="#EBEBEB", linewidth=1)
ax.axhline(zs, color="#EBEBEB", linewidth=1)
ax.set_title('10个公开数据集上的平均结果', fontsize=14, fontfamily='sans-serif', fontstyle='italic',
             fontproperties=font_set)
ax.set_ylim(val_bot, val_top)

ax.plot(
    0, zs,
    marker="*",
    markersize=MS * 1.5,
    color=COLORS["zs"],
    alpha=ALPHA
)
ax.plot(
    shots, CoOp_v16_end,
    marker="o",
    markersize=MS,
    color=COLORS["CoOp_v16_end"],
    label="CLIP + CoOp (M=16, end)",
    linestyle="--",
    alpha=ALPHA
)
ax.plot(
    shots, CoOp_v16_mid,
    marker="o",
    markersize=MS,
    color=COLORS["CoOp_v16_mid"],
    label="CLIP + CoOp (M=16, mid)",
    linestyle="-.",
    alpha=ALPHA
)
ax.plot(
    shots, PTVE,
    marker="o",
    markersize=MS,
    color=COLORS["PTVE"],
    label="PTVE",
    alpha=ALPHA
)
ax.plot(
    shots, linear,
    marker="o",
    markersize=MS,
    color=COLORS["linear"],
    label="Linear probe CLIP",
    linestyle="dotted",
    alpha=ALPHA
)

# ax.text(-0.5, zs - diff * 0.11, "Zero-shot\nCLIP", color=COLORS["zs"])
ax.text(-0.5, zs - diff * 0.05, "CLIP零样本评估", color=COLORS["zs"], fontproperties=font_set_1)
ax.legend(loc="lower right")

# fig.savefig(f"{save_dir}/average.pdf", bbox_inches="tight")
fig.savefig(f"{save_dir}/average.png", bbox_inches="tight")
