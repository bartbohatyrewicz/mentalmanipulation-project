#!/usr/bin/env python3

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from transformers import AutoTokenizer, DebertaV2Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import MentalManipMultiTask


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class MentalManipPredictor:
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "configs/config.yaml",
        device: str = None
    ):

        self.config = load_config(config_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model from: {model_path}")
        self.model = MentalManipMultiTask.load(model_path, device=self.device)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = DebertaV2Tokenizer.from_pretrained("deberta")

        self.techniques = self.config['techniques']
        self.vulnerabilities = self.config['vulnerabilities']
        
        print(f"Predictor ready (device: {self.device})")
    
    @torch.no_grad()
    def predict(
        self,
        dialogue: str,
        threshold: float = 0.5
    ) -> dict:

        inputs = self.tokenizer(
            dialogue,
            truncation=True,
            max_length=self.config['data']['max_length'],
            padding="max_length",
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        manip_probs = torch.softmax(outputs["manip_logits"], dim=-1)
        is_manipulative = manip_probs[0, 1].item() > 0.5
        manip_confidence = manip_probs[0, 1].item()

        tech_probs = torch.sigmoid(outputs["tech_logits"]).cpu().numpy()[0]
        techniques = [
            self.techniques[i]
            for i in range(len(self.techniques))
            if tech_probs[i] >= threshold
        ]

        vuln_probs = torch.sigmoid(outputs["vuln_logits"]).cpu().numpy()[0]
        vulnerabilities = [
            self.vulnerabilities[i]
            for i in range(len(self.vulnerabilities))
            if vuln_probs[i] >= threshold
        ]
        
        return {
            "manipulative": is_manipulative,
            "confidence": manip_confidence,
            "techniques": techniques,
            "techniques_scores": {
                self.techniques[i]: float(tech_probs[i])
                for i in range(len(self.techniques))
            },
            "vulnerabilities": vulnerabilities,
            "vulnerabilities_scores": {
                self.vulnerabilities[i]: float(vuln_probs[i])
                for i in range(len(self.vulnerabilities))
            }
        }
    
    def predict_batch(
        self,
        dialogues: list,
        threshold: float = 0.5
    ) -> list:

        return [self.predict(d, threshold) for d in dialogues]


def format_prediction(pred: dict) -> str:
    lines = []

    status = "MANIPULATIVE" if pred["manipulative"] else "NOT MANIPULATIVE"
    lines.append(f"\n{status} (confidence: {pred['confidence']:.2%})")

    if pred["techniques"]:
        lines.append(f"\nTechniques detected:")
        for tech in pred["techniques"]:
            score = pred["techniques_scores"][tech]
            lines.append(f"  • {tech} ({score:.2%})")

    if pred["vulnerabilities"]:
        lines.append(f"\nVulnerabilities exploited:")
        for vuln in pred["vulnerabilities"]:
            score = pred["vulnerabilities_scores"][vuln]
            lines.append(f"  • {vuln} ({score:.2%})")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run inference on dialogues")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single dialogue text to analyze"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File with dialogues (one per line)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for multi-label classification"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    args = parser.parse_args()

    predictor = MentalManipPredictor(
        model_path=args.model,
        config_path=args.config
    )
    
    if args.text:

        print("INPUT DIALOGUE:")
        print(args.text)
        
        pred = predictor.predict(args.text, args.threshold)
        print("PREDICTION:")
        print(format_prediction(pred))
    
    elif args.file:
        with open(args.file, 'r') as f:
            dialogues = [line.strip() for line in f if line.strip()]
        
        print(f"\nProcessing {len(dialogues)} dialogues...")
        
        for i, dialogue in enumerate(dialogues):
            print(f"DIALOGUE {i+1}:")
            print(dialogue[:200] + "..." if len(dialogue) > 200 else dialogue)
            
            pred = predictor.predict(dialogue, args.threshold)
            print(format_prediction(pred))
    
    elif args.interactive:
        print("INTERACTIVE MODE:\n")
        print("Enter dialogues to analyze. Type 'quit' to exit.\n")
        
        while True:
            dialogue = input("\nDialogue: ").strip()
            
            if dialogue.lower() in ['quit', 'exit', 'q']:
                break
            
            if not dialogue:
                continue
            
            pred = predictor.predict(dialogue, args.threshold)
            print(format_prediction(pred))
    
    else:
        print("Please provide --text, --file, or --interactive")
        parser.print_help()


if __name__ == "__main__":
    main()
