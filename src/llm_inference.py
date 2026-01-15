import re
import json
import torch
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score


class LLMClassifier:

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit: bool = True,
        techniques: Optional[List[str]] = None,
        vulnerabilities: Optional[List[str]] = None,
        device: str = "auto"
    ):

        print(f"Loading LLM: {model_id}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        load_kwargs = {
            "device_map": device,
            "trust_remote_code": True
        }
        
        if load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            print("Using 4-bit quantization")
        
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()
        
        self.techniques = techniques or []
        self.vulnerabilities = vulnerabilities or []
        
        print(f"LLM loaded successfully")
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        markers = list(re.finditer(r'JSON:\s*', text))
        if not markers:
            try:
                match = re.search(r'\{[^}]+\}', text)
                if match:
                    return json.loads(match.group())
            except:
                pass
            return None
        
        start = markers[-1].end()
        json_str = text[start:].strip()

        try:
            return json.loads(json_str)
        except:
            match = re.search(r'\{[^}]+\}', json_str)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
        return None
    
    def _build_prompt(
        self,
        dialogue: str,
        examples: Optional[List[Dict]] = None
    ) -> str:

        tech_list = ", ".join(self.techniques) if self.techniques else "various techniques"
        vuln_list = ", ".join(self.vulnerabilities) if self.vulnerabilities else "various vulnerabilities"
        
        system = f"""You are an expert NLP classifier for psychological manipulation detection.
Analyze dialogues and output JSON with:
- "manipulative": 1 if manipulation present, 0 if not
- "techniques": list of techniques from [{tech_list}]
- "vulnerabilities": list of vulnerabilities from [{vuln_list}]
- "confidence": your confidence (0.0-1.0)

Only use labels from the provided lists. Output valid JSON only."""

        prompt = system + "\n\n"

        if examples:
            prompt += "Examples:\n\n"
            for ex in examples:
                prompt += f'Dialogue: """{ex["dialogue"]}"""\n'
                prompt += f'JSON: {json.dumps(ex["json"])}\n\n'

        prompt += f'Dialogue: """{dialogue}"""\nJSON:'
        
        return prompt
    
    def predict(
        self,
        dialogue: str,
        examples: Optional[List[Dict]] = None,
        max_new_tokens: int = 64
    ) -> Dict[str, Any]:

        prompt = self._build_prompt(dialogue, examples)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = self._extract_json(decoded)
        
        if result is None:
            result = {
                "manipulative": 0,
                "techniques": [],
                "vulnerabilities": [],
                "confidence": 0.0
            }
        
        return result
    
    def predict_binary(
        self,
        dialogue: str,
        examples: Optional[List[Dict]] = None
    ) -> int:

        result = self.predict(dialogue, examples)
        return int(result.get("manipulative", 0))


def get_few_shot_examples(dataset, n_manip: int = 3, n_non_manip: int = 1) -> List[Dict]:

    examples = []

    manip_samples = [x for x in dataset if x['manipulative'] == 1][:n_manip]
    for sample in manip_samples:
        examples.append({
            "dialogue": sample['dialogue'][:500],  # Truncate long dialogues
            "json": {
                "manipulative": 1,
                "techniques": sample.get('techniques', [])[:3],
                "vulnerabilities": sample.get('vulnerabilities', [])[:2],
                "confidence": 0.9
            }
        })

    non_manip_samples = [x for x in dataset if x['manipulative'] == 0][:n_non_manip]
    for sample in non_manip_samples:
        examples.append({
            "dialogue": sample['dialogue'][:500],
            "json": {
                "manipulative": 0,
                "techniques": [],
                "vulnerabilities": [],
                "confidence": 0.9
            }
        })
    
    return examples


def evaluate_llm(
    classifier: LLMClassifier,
    dataset,
    n_samples: int = 20,
    use_few_shot: bool = False,
    few_shot_examples: Optional[List[Dict]] = None
) -> Dict[str, float]:

    indices = list(range(min(n_samples, len(dataset))))
    
    y_true = []
    y_pred = []
    
    examples = few_shot_examples if use_few_shot else None
    mode = "Few-shot" if use_few_shot else "Zero-shot"
    
    print(f"\nEvaluating LLM ({mode}) on {len(indices)} samples...")
    
    for i in tqdm(indices):
        sample = dataset[i]
        dialogue = sample['dialogue']
        true_label = int(sample['manipulative'])
        
        pred_label = classifier.predict_binary(dialogue, examples)
        
        y_true.append(true_label)
        y_pred.append(pred_label)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    
    print(f"\n{mode} Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    return {
        "accuracy": acc,
        "f1_score": f1,
        "y_true": y_true,
        "y_pred": y_pred
    }
