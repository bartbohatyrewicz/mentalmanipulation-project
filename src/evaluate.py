import torch
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate_multilabel(model, dataset, threshold: float = 0.5,
                        batch_size: int = 16, device: Optional[str] = None
                        ) -> Dict[str, Any]:

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    
    loader = DataLoader(dataset, batch_size=batch_size)
    
    tech_logits_all, vuln_logits_all = [], []
    tech_true_all, vuln_true_all = [], []
    
    for batch in tqdm(loader, desc="Evaluating multi-label"):
        labels_tech = batch.pop("labels_tech")
        labels_vuln = batch.pop("labels_vuln")
        batch.pop("label_manip", None)

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        
        tech_logits_all.append(outputs["tech_logits"].cpu())
        vuln_logits_all.append(outputs["vuln_logits"].cpu())
        tech_true_all.append(labels_tech)
        vuln_true_all.append(labels_vuln)

    tech_logits = torch.cat(tech_logits_all).numpy()
    vuln_logits = torch.cat(vuln_logits_all).numpy()
    tech_true = torch.cat(tech_true_all).numpy().astype(int)
    vuln_true = torch.cat(vuln_true_all).numpy().astype(int)

    tech_prob = 1 / (1 + np.exp(-tech_logits))
    vuln_prob = 1 / (1 + np.exp(-vuln_logits))
    
    tech_pred = (tech_prob >= threshold).astype(int)
    vuln_pred = (vuln_prob >= threshold).astype(int)
    
    results = {
        "tech_micro_f1": f1_score(tech_true, tech_pred, average="micro", zero_division=0),
        "tech_macro_f1": f1_score(tech_true, tech_pred, average="macro", zero_division=0),
        "vuln_micro_f1": f1_score(vuln_true, vuln_pred, average="micro", zero_division=0),
        "vuln_macro_f1": f1_score(vuln_true, vuln_pred, average="macro", zero_division=0),
        "tech_predictions": tech_pred,
        "tech_true": tech_true,
        "vuln_predictions": vuln_pred,
        "vuln_true": vuln_true,
    }
    
    return results


@torch.no_grad()
def evaluate_binary(model, dataset, batch_size: int = 16,
                    device: Optional[str] = None
                    ) -> Dict[str, Any]:

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    
    loader = DataLoader(dataset, batch_size=batch_size)
    
    all_preds = []
    all_true = []
    
    for batch in tqdm(loader, desc="Evaluating binary"):
        batch.pop("labels_tech")
        batch.pop("labels_vuln")
        label_manip = batch.pop("label_manip")

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)

        preds = torch.argmax(outputs["manip_logits"], dim=-1).cpu()
        all_preds.extend(preds.tolist())
        all_true.extend(label_manip.tolist())

    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average='binary')
    cm = confusion_matrix(all_true, all_preds)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        all_true, all_preds, average=None, zero_division=0
    )
    
    results = {
        "accuracy": acc,
        "f1_score": f1,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_per_class": f1_per_class,
        "support": support,
        "y_true": all_true,
        "y_pred": all_preds
    }
    
    return results


def evaluate_model(model, dataset,batch_size: int = 16,
                   device: Optional[str] = None
                    ) -> Dict[str, Any]:

    print("- EVALUATING MODEL -")

    print("\n- Binary Classification -")
    binary_results = evaluate_binary(model, dataset, batch_size, device)
    
    print(f"Accuracy: {binary_results['accuracy']:.4f}")
    print(f"F1 Score: {binary_results['f1_score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        binary_results['y_true'],
        binary_results['y_pred'],
        target_names=["Not Manipulative", "Manipulative"]
    ))

    print("\nMulti-label Classification")
    multilabel_results = evaluate_multilabel(model, dataset, batch_size=batch_size, device=device)
    
    print(f"Techniques - Micro F1: {multilabel_results['tech_micro_f1']:.4f}")
    print(f"Techniques - Macro F1: {multilabel_results['tech_macro_f1']:.4f}")
    print(f"Vulnerabilities - Micro F1: {multilabel_results['vuln_micro_f1']:.4f}")
    print(f"Vulnerabilities - Macro F1: {multilabel_results['vuln_macro_f1']:.4f}")
    
    return {
        "binary": binary_results,
        "multilabel": multilabel_results
    }


def evaluate_per_technique(
    model,
    dataset,
    technique_names: List[str],
    threshold: float = 0.5,
    batch_size: int = 16,
    device: Optional[str] = None
) -> Dict[str, Dict[str, float]]:

    results = evaluate_multilabel(model, dataset, threshold, batch_size, device)
    
    tech_pred = results["tech_predictions"]
    tech_true = results["tech_true"]
    
    per_technique = {}
    
    for i, name in enumerate(technique_names):
        precision, recall, f1, _ = precision_recall_fscore_support(
            tech_true[:, i],
            tech_pred[:, i],
            average='binary',
            zero_division=0
        )
        
        support = int(tech_true[:, i].sum())
        
        per_technique[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }
    
    return per_technique
