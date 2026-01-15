import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Optional, Any


class MentalManipMultiTask(nn.Module):
    def __init__(
        self,
        backbone: str,
        n_techniques: int,
        n_vulnerabilities: int,
        use_manip_head: bool = True,
        dropout: float = 0.1
    ):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(backbone)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.tech_head = nn.Linear(hidden_size, n_techniques)
        self.vuln_head = nn.Linear(hidden_size, n_vulnerabilities)
        
        self.use_manip_head = use_manip_head
        if use_manip_head:
            self.manip_head = nn.Linear(hidden_size, 2)

        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_ce = nn.CrossEntropyLoss()

        self.hparams = {
            'backbone': backbone,
            'n_techniques': n_techniques,
            'n_vulnerabilities': n_vulnerabilities,
            'use_manip_head': use_manip_head,
            'dropout': dropout
        }

        @property
        def config(self):
            return self.encoder.config
    
    def _pool(
        self,
        encoder_output: Any,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:

        if hasattr(encoder_output, "pooler_output") and encoder_output.pooler_output is not None:
            return encoder_output.pooler_output

        last_hidden = encoder_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        
        return summed / counts
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_tech: Optional[torch.Tensor] = None,
        labels_vuln: Optional[torch.Tensor] = None,
        label_manip: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        pooled = self._pool(encoder_output, attention_mask)
        pooled = self.dropout(pooled)

        tech_logits = self.tech_head(pooled)
        vuln_logits = self.vuln_head(pooled)
        
        output = {
            "tech_logits": tech_logits,
            "vuln_logits": vuln_logits
        }
        
        if self.use_manip_head:
            manip_logits = self.manip_head(pooled)
            output["manip_logits"] = manip_logits

        if labels_tech is not None or labels_vuln is not None:
            loss = torch.tensor(0.0, device=input_ids.device)
            
            if labels_tech is not None:
                loss = loss + self.loss_bce(tech_logits, labels_tech)
            
            if labels_vuln is not None:
                loss = loss + self.loss_bce(vuln_logits, labels_vuln)
            
            if self.use_manip_head and label_manip is not None:
                loss = loss + self.loss_ce(manip_logits, label_manip)
            
            output["loss"] = loss
        
        return output
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'hparams': self.hparams
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'MentalManipMultiTask':
        checkpoint = torch.load(path, map_location=device)
        hparams = checkpoint['hparams']
        
        model = cls(
            backbone=hparams['backbone'],
            n_techniques=hparams['n_techniques'],
            n_vulnerabilities=hparams['n_vulnerabilities'],
            use_manip_head=hparams['use_manip_head'],
            dropout=hparams['dropout']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        
        return model


def create_model(config: Dict[str, Any]) -> MentalManipMultiTask:
    model = MentalManipMultiTask(
        backbone=config['model']['backbone'],
        n_techniques=len(config['techniques']),
        n_vulnerabilities=len(config['vulnerabilities']),
        use_manip_head=config['model']['use_manip_head'],
        dropout=config['model']['dropout']
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created: {total_params:,} total params, {trainable_params:,} trainable")
    
    return model
