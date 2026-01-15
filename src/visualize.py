import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = ["Not Manip", "Manipulative"],
    title: str = "Binary Manipulation – Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_confusion_matrix_normalized(
    cm: np.ndarray,
    labels: List[str] = ["Not Manip", "Manipulative"],
    title: str = "Normalized Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(cm_norm)
    plt.colorbar()
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.title(title)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return plt.gcf()


def plot_class_metrics(
    binary_results: Dict[str, Any],
    title: str = "Binary Head – Class-wise Metrics",
    save_path: Optional[str] = None
) -> plt.Figure:

    metrics = {
        "Precision": binary_results["precision"],
        "Recall": binary_results["recall"],
        "F1": binary_results["f1_per_class"]
    }

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(x - width/2, [metrics[m][0] for m in metrics], width, label="Not Manip")
    ax.bar(x + width/2, [metrics[m][1] for m in metrics], width, label="Manipulative")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_multilabel_metrics(
    multilabel_results: Dict[str, float],
    title: str = "Multilabel Performance – Micro vs Macro F1",
    save_path: Optional[str] = None
) -> plt.Figure:

    labels = ["Techniques", "Vulnerabilities"]
    micro_f1 = [multilabel_results["tech_micro_f1"], multilabel_results["vuln_micro_f1"]]
    macro_f1 = [multilabel_results["tech_macro_f1"], multilabel_results["vuln_macro_f1"]]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, micro_f1, width, label="Micro F1")
    plt.bar(x + width/2, macro_f1, width, label="Macro F1")

    plt.xticks(x, labels)
    plt.ylim(0, 0.4)
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return plt.gcf()


def plot_summary(
    binary_results: Dict[str, Any],
    multilabel_results: Dict[str, float],
    title: str = "Model Performance Summary",
    save_path: Optional[str] = None
) -> plt.Figure:

    metrics = {
        "Binary F1": binary_results["f1_score"],
        "Tech Micro F1": multilabel_results["tech_micro_f1"],
        "Tech Macro F1": multilabel_results["tech_macro_f1"],
        "Vuln Micro F1": multilabel_results["vuln_micro_f1"],
        "Vuln Macro F1": multilabel_results["vuln_macro_f1"]
    }

    plt.figure(figsize=(8, 4))
    plt.bar(metrics.keys(), metrics.values())
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return plt.gcf()


def plot_per_technique_f1(
    per_technique: Dict[str, Dict[str, float]],
    title: str = "F1 Score per Manipulation Technique",
    save_path: Optional[str] = None
) -> plt.Figure:

    techniques = list(per_technique.keys())
    f1_scores = [per_technique[t]["f1"] for t in techniques]
    supports = [per_technique[t]["support"] for t in techniques]

    sorted_indices = np.argsort(f1_scores)[::-1]
    techniques = [techniques[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    supports = [supports[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(techniques, f1_scores)
    
    ax.set_xlabel("F1 Score")
    ax.set_title(title)
    ax.set_xlim(0, max(f1_scores) * 1.2 if max(f1_scores) > 0 else 0.5)

    for i, (bar, support) in enumerate(zip(bars, supports)):
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'n={support}',
            ha='left', va='center',
            fontsize=9
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_all_plots(
    results: Dict[str, Any],
    technique_names: List[str],
    output_dir: str = "outputs/plots"
) -> None:

    os.makedirs(output_dir, exist_ok=True)
    
    binary = results["binary"]
    multilabel = results["multilabel"]
    
    print("\nGenerating plots...")

    plot_confusion_matrix(
        binary["confusion_matrix"],
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )

    plot_confusion_matrix_normalized(
        binary["confusion_matrix"],
        save_path=os.path.join(output_dir, "confusion_matrix_normalized.png")
    )

    plot_class_metrics(
        binary,
        save_path=os.path.join(output_dir, "class_metrics.png")
    )

    plot_multilabel_metrics(
        multilabel,
        save_path=os.path.join(output_dir, "multilabel_metrics.png")
    )

    plot_summary(
        binary,
        multilabel,
        save_path=os.path.join(output_dir, "summary.png")
    )
    
    print(f"\nAll plots saved to {output_dir}/")
    plt.close('all')
