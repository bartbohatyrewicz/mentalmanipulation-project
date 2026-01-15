#!/usr/bin/env python3

import os
import sys
import argparse
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import prepare_data
from src.model import MentalManipMultiTask
from src.evaluate import evaluate_model, evaluate_per_technique
from src.visualize import generate_all_plots, plot_per_technique_f1


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Evaluate MentalManip model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval",
        help="Directory for evaluation outputs"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    args = parser.parse_args()

    print("- MentalManip - Model Evaluation -")
    
    config = load_config(args.config)

    print("\nPreparing data...")
    data = prepare_data(config)

    print(f"\nLoading model from: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MentalManipMultiTask.load(args.model, device=device)
    model.to(device)
    model.eval()

    print("- EVALUATION RESULTS -")
    
    results = evaluate_model(
        model=model,
        dataset=data['test_enc'],
        device=device
    )

    per_tech = evaluate_per_technique(
        model=model,
        dataset=data['test_enc'],
        technique_names=config['techniques'],
        device=device
    )
    
    print("\nPer-Technique F1 Scores")
    for tech, metrics in sorted(per_tech.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"  {tech}: {metrics['f1']:.3f} (n={metrics['support']})")

    if not args.no_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        
        generate_all_plots(
            results=results,
            technique_names=config['techniques'],
            output_dir=args.output_dir
        )
        
        plot_per_technique_f1(
            per_tech,
            save_path=os.path.join(args.output_dir, "per_technique_f1.png")
        )
        
        print(f"\nPlots saved to: {args.output_dir}")

    results_file = os.path.join(args.output_dir, "results.txt")
    with open(results_file, 'w') as f:
        f.write("MentalManip Evaluation Results\n")
        
        f.write("Binary Classification:\n")
        f.write(f"  Accuracy: {results['binary']['accuracy']:.4f}\n")
        f.write(f"  F1 Score: {results['binary']['f1_score']:.4f}\n\n")
        
        f.write("Multi-label Classification:\n")
        f.write(f"  Techniques Micro F1: {results['multilabel']['tech_micro_f1']:.4f}\n")
        f.write(f"  Techniques Macro F1: {results['multilabel']['tech_macro_f1']:.4f}\n")
        f.write(f"  Vulnerabilities Micro F1: {results['multilabel']['vuln_micro_f1']:.4f}\n")
        f.write(f"  Vulnerabilities Macro F1: {results['multilabel']['vuln_macro_f1']:.4f}\n\n")
        
        f.write("Per-Technique F1:\n")
        for tech, metrics in sorted(per_tech.items(), key=lambda x: x[1]['f1'], reverse=True):
            f.write(f"  {tech}: {metrics['f1']:.3f} (n={metrics['support']})\n")
    
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
