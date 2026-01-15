import numpy as np
from typing import Dict, List, Any, Optional
from datasets import load_dataset, Dataset, ClassLabel, Value
from transformers import AutoTokenizer, DebertaV2Tokenizer


def load_mentalmanip(dataset_name: str = "audreyeleven/MentalManip",
                    config_name: str = "mentalmanip_detailed"
                     ) -> Dataset:

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, config_name)
    print(f"Loaded {len(dataset['train'])} samples")
    return dataset


def aggregate_annotations(dataset: Dataset) -> Dataset:

    print("Aggregating annotations...")
    
    new_data = []
    data = dataset["train"]
    
    for row in data:
        manipulative = 1 if (
            row.get('manipulative_1') == 1 or
            row.get('manipulative_2') == 1 or
            row.get('manipulative_3') == 1
        ) else 0

        techniques = set()
        for t in [row.get('technique_1'), row.get('technique_2'), row.get('technique_3')]:
            if t:
                for tech in t.split(','):
                    tech = tech.strip()
                    if tech:
                        techniques.add(tech)

        vulnerabilities = set()
        for v in [row.get('vulnerability_1'), row.get('vulnerability_2'), row.get('vulnerability_3')]:
            if v:
                for vuln in v.split(','):
                    vuln = vuln.strip()
                    if vuln:
                        vulnerabilities.add(vuln)
        
        new_data.append({
            'dialogue': row['dialogue'],
            'manipulative': manipulative,
            'techniques': list(techniques) if techniques else [],
            'vulnerabilities': list(vulnerabilities) if vulnerabilities else []
        })
    
    new_dataset = Dataset.from_list(new_data)

    manip_count = sum(1 for x in new_data if x['manipulative'] == 1)
    tech_count = sum(1 for x in new_data if len(x['techniques']) > 0)
    vuln_count = sum(1 for x in new_data if len(x['vulnerabilities']) > 0)
    
    print(f"Total samples: {len(new_data)}")
    print(f"Manipulative: {manip_count}")
    print(f"With techniques: {tech_count}")
    print(f"With vulnerabilities: {vuln_count}")
    
    return new_dataset


def split_dataset(dataset: Dataset, test_size: float = 0.2,
                    seed: int = 42) -> Dict[str, Dataset]:

    print(f"Splitting dataset: {1-test_size:.0%} train, {test_size:.0%} test")

    dataset = dataset.cast_column("manipulative", Value("int64"))
    dataset = dataset.cast_column(
        "manipulative", 
        ClassLabel(num_classes=2, names=["not_manipulative", "manipulative"])
    )
    
    split = dataset.train_test_split(
        test_size=test_size,
        seed=seed,
        stratify_by_column="manipulative"
    )
    
    print(f"Train size: {len(split['train'])}")
    print(f"Test size: {len(split['test'])}")
    
    return split


class DataEncoder:

    def __init__(
        self,
        tokenizer_name: str,
        techniques: List[str],
        vulnerabilities: List[str],
        max_length: int = 512
    ):

        if "deberta" in tokenizer_name.lower():
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.max_length = max_length

        self.techniques = techniques
        self.vulnerabilities = vulnerabilities
        self.tech2id = {t: i for i, t in enumerate(techniques)}
        self.vuln2id = {v: i for i, v in enumerate(vulnerabilities)}
        
        print(f"Encoder initialized: {len(techniques)} techniques, {len(vulnerabilities)} vulnerabilities")
    
    def _ensure_list(self, x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            return [s.strip() for s in x.split(",") if s.strip()]
        return []
    
    def encode_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        enc = self.tokenizer(
            batch["dialogue"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        
        n = len(batch["dialogue"])

        y_tech = np.zeros((n, len(self.techniques)), dtype=np.float32)
        y_vuln = np.zeros((n, len(self.vulnerabilities)), dtype=np.float32)
        
        for i in range(n):
            for t in self._ensure_list(batch["techniques"][i]):
                if t in self.tech2id:
                    y_tech[i, self.tech2id[t]] = 1.0
            
            for v in self._ensure_list(batch["vulnerabilities"][i]):
                if v in self.vuln2id:
                    y_vuln[i, self.vuln2id[v]] = 1.0
        
        enc["labels_tech"] = y_tech
        enc["labels_vuln"] = y_vuln
        enc["label_manip"] = np.array(batch["manipulative"], dtype=np.int64)
        
        return enc
    
    def encode_dataset(self, dataset: Dataset) -> Dataset:
        encoded = dataset.map(
            self.encode_batch,
            batched=True,
            remove_columns=dataset.column_names
        )

        encoded.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels_tech", "labels_vuln", "label_manip"]
        )
        
        return encoded


def prepare_data(config: Dict[str, Any]) -> Dict[str, Any]:
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

    encoder = DataEncoder(
        tokenizer_name=config['model']['backbone'],
        techniques=config['techniques'],
        vulnerabilities=config['vulnerabilities'],
        max_length=config['data']['max_length']
    )

    print("Encoding...")
    train_enc = encoder.encode_dataset(split['train'])
    test_enc = encoder.encode_dataset(split['test'])
    
    return {
        'train_enc': train_enc,
        'test_enc': test_enc,
        'train_raw': split['train'],
        'test_raw': split['test'],
        'encoder': encoder
    }
