import itertools
import pickle
import os
import matplotlib.pyplot as plt
from metrics_utils import *


# Function to plot boxplots with p-value annotations
def plot_boxplot_with_p_value(data, title, ylabel, categories, background = False, outdir = None):
    fontsize = 20
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    plt.boxplot([data[bag][title] for bag in BAGNAMES], labels=[categories[bag] for bag in BAGNAMES])
    # plt.xlim(left=0)
    plt.title(f"{title}", fontdict={"fontsize": fontsize})
    plt.ylabel(ylabel, fontdict={"fontsize": fontsize})
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(fontsize=fontsize)
    # plt.ylim(0, 7.6 * 1.05)
    plt.ylim(0, 1.1)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=fontsize)  # Adjust `num=6` as needed

    if background:
        # Define bands and labels
        bands = [
            (0, 0.5, "Intimate", "tab:red"),
            (0.5, 1.2, "Personal", "tab:orange"),
            (1.2, 3.6, "Social", "tab:blue"),
            (3.6, 7.6, "Public", "tab:green"),
        ]

        # Add colored bands and labels
        for lower, upper, label, color in bands:
            ax.axhspan(lower, upper, color=color, alpha=0.3)
            ax.text(plt.xlim()[0] + 0.05, (lower + upper) / 2, label, 
                    va="center", ha="left", fontsize=fontsize-1, color="black")  # Left-aligned
        
    if outdir is not None:
        plt.savefig(os.path.join(outdir, f'{title.lower().replace(" ", "_")}_boxplot.png'))
    else:
        plt.show()
    plt.tight_layout()
    plt.close()

INDIR = '/home/hrisim/ros_ws/src/HRISim/postprocessing/hrisim_postprocess/csv/'
BAGNAMES = ['REASONING_NONCAUSAL_POSTER', 'REASONING_CAUSAL_POSTER', 'REASONING_NONCAUSAL_BUFFET', 'REASONING_CAUSAL_BUFFET']
CATEGORIES = {'REASONING_NONCAUSAL_POSTER': 'baseline-S1', 'REASONING_CAUSAL_POSTER': 'causal-S1', 'REASONING_NONCAUSAL_BUFFET': 'baseline-S2', 'REASONING_CAUSAL_BUFFET': 'causal-S2'}
OUTDIR = '/home/hrisim/ros_ws/src/HRISim/postprocessing/hrisim_postprocess/results/'
os.makedirs(OUTDIR, exist_ok=True)

# Initialize aggregated data structures
proxemics_metrics = {bagname: {"Human-Robot Proxemic Compliance": []} for bagname in BAGNAMES}
completion_metrics = {bagname: {"Completion Percentage": []} for bagname in BAGNAMES}

# Load metrics for each bag
for bagname in BAGNAMES:
    metrics_path = os.path.join(INDIR, bagname, "metrics.pkl")
    with open(metrics_path, 'rb') as pkl_file:
        METRICS = pickle.load(pkl_file)
    for task in METRICS.keys():
        try:
            if isinstance(int(task), int):
                proxemics_metrics[bagname]["Human-Robot Proxemic Compliance"].extend(
                    value for value in itertools.chain(*METRICS[task]['agent_distances'].values()) if value < 7.6
                )
                completion_metrics[bagname]["Completion Percentage"].append(METRICS[task]['completion_percentage'])
        except ValueError:
            continue

# Plot all metrics as boxplot with p-values
metrics = {
    # "proxemics_metrics": ["Human-Robot Proxemic Compliance"],
    "completion_metrics": ["Completion Percentage"],
}

for metric_type, metric_list in metrics.items():
    print(metric_type)
    for metric_name in metric_list:
        data = eval(metric_type)  # Get the data dictionary dynamically
        plot_boxplot_with_p_value(data, metric_name, "m", CATEGORIES, background = metric_type == "proxemics_metrics", outdir=OUTDIR)