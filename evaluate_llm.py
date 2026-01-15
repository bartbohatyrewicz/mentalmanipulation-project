#!/usr/bin/env python3

import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import load_mentalmanip, aggregate_annotations, split_dataset
from src.llm_inference import LLMClassifier, get_few_shot_examples, evaluate_llm


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM baselines")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default from config)"
    )
    parser.add_argument(
        "--zero-shot-only",
        action="store_true",
        help="Only run zero-shot evaluation"
    )
    parser.add_argument(
        "--few-shot-only",
        action="store_true",
        help="Only run few-shot evaluation"
    )
    args = parser.parse_args()

    print("- MentalManip - LLM Baseline Evaluation -")
    
    config = load_config(args.config)
    n_samples = args.n_samples or config['llm']['num_eval_samples']

    print("\nLoading data...")
    raw_dataset = load_mentalmanip(
        config['data']['dataset_name'],
        config['data']['dataset_config']
    )
    aggregated = aggregate_annotations(raw_dataset)
    split = split_dataset(
        aggregated,
        test_size=config['data']['test_size'],
        seed=config['data']['seed']
    )

    print("- LOADING LLM -")
    
    classifier = LLMClassifier(
        model_id=config['llm']['model_id'],
        load_in_4bit=config['llm']['load_in_4bit'],
        techniques=config['techniques'],
        vulnerabilities=config['vulnerabilities']
    )

    few_shot_examples = get_few_shot_examples(split['train'], n_manip=3, n_non_manip=1)
    
    results = {}

    if not args.few_shot_only:
        print("- ZERO-SHOT EVALUATION -")
        
        results['zero_shot'] = evaluate_llm(
            classifier=classifier,
            dataset=split['test'],
            n_samples=n_samples,
            use_few_shot=False
        )

    if not args.zero_shot_only:
        print("- FEW-SHOT EVALUATION -")
        
        results['few_shot'] = evaluate_llm(
            classifier=classifier,
            dataset=split['test'],
            n_samples=n_samples,
            use_few_shot=True,
            few_shot_examples=few_shot_examples
        )

    print("- SUMMARY -")
    
    print(f"\nEvaluated on {n_samples} samples\n")
    
    if 'zero_shot' in results:
        print(f"Zero-shot:")
        print(f"  Accuracy: {results['zero_shot']['accuracy']:.4f}")
        print(f"  F1 Score: {results['zero_shot']['f1_score']:.4f}")
    
    if 'few_shot' in results:
        print(f"\nFew-shot (4 examples):")
        print(f"  Accuracy: {results['few_shot']['accuracy']:.4f}")
        print(f"  F1 Score: {results['few_shot']['f1_score']:.4f}")

    os.makedirs("outputs", exist_ok=True)
    results_file = "outputs/llm_results.txt"

    with open(results_file, 'w') as f:
        f.write("LLM Baseline Results\n")
        f.write(f"Model: {config['llm']['model_id']}\n")
        f.write(f"Samples evaluated: {n_samples}\n\n")
        
        if 'zero_shot' in results:
            f.write("Zero-shot:\n")
            f.write(f"  Accuracy: {results['zero_shot']['accuracy']:.4f}\n")
            f.write(f"  F1 Score: {results['zero_shot']['f1_score']:.4f}\n\n")
        
        if 'few_shot' in results:
            f.write("Few-shot:\n")
            f.write(f"  Accuracy: {results['few_shot']['accuracy']:.4f}\n")
            f.write(f"  F1 Score: {results['few_shot']['f1_score']:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
