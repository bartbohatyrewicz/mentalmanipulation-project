#!/usr/bin/env python3

import os
import sys
import argparse
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import prepare_data
from src.model import create_model
from src.trainer import train_model
from src.evaluate import evaluate_model, evaluate_per_technique
from src.visualize import generate_all_plots, plot_per_technique_f1


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train MentalManip model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation after training"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    args = parser.parse_args()

    print("MentalManip - Psychological Manipulation Detection")
    print(f"\nLoading config: {args.config}")
    
    config = load_config(args.config)

    print(f"\nModel: {config['model']['backbone']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Batch size: {config['training']['batch_size']} x {config['training']['gradient_accumulation_steps']}")
    print(f"Epochs: {config['training']['num_epochs']}")

    print("- PREPARING DATA -")
    
    data = prepare_data(config)

    print("- CREATING MODEL -")
    
    model = create_model(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    trainer = train_model(
        model=model,
        train_dataset=data['train_enc'],
        eval_dataset=data['test_enc'],
        tokenizer=data['encoder'].tokenizer,
        config=config
    )

    if not args.no_eval:
        print("- EVALUATION -")
        
        results = evaluate_model(
            model=trainer.model,
            dataset=data['test_enc'],
            device=device
        )

        per_tech = evaluate_per_technique(
            model=trainer.model,
            dataset=data['test_enc'],
            technique_names=config['techniques'],
            device=device
        )
        
        print("\nPer-Technique F1 Scores")
        for tech, metrics in sorted(per_tech.items(), key=lambda x: x[1]['f1'], reverse=True):
            print(f"  {tech}: {metrics['f1']:.3f} (n={metrics['support']})")

        if not args.no_plots:
            generate_all_plots(
                results=results,
                technique_names=config['techniques'],
                output_dir=config['output']['plots_dir']
            )

            plot_per_technique_f1(
                per_tech,
                save_path=os.path.join(config['output']['plots_dir'], "per_technique_f1.png")
            )

    print("- TRAINING COMPLETE -")
    print(f"\nModel saved to: {config['output']['model_dir']}")
    print(f"Logs saved to: {config['output']['logs_dir']}")
    if not args.no_plots:
        print(f"Plots saved to: {config['output']['plots_dir']}")
    
    print("\nTo view TensorBoard logs:")
    print(f"  tensorboard --logdir {config['output']['logs_dir']}")


if __name__ == "__main__":
    main()
