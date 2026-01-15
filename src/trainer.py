import os
from typing import Dict, Any, Optional
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from .model import MentalManipMultiTask


class MultiTaskTrainer(Trainer):

    def compute_loss(
        self,
        model: MentalManipMultiTask,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs
    ):

        labels_tech = inputs.pop("labels_tech")
        labels_vuln = inputs.pop("labels_vuln")
        label_manip = inputs.pop("label_manip", None)

        outputs = model(
            **inputs,
            labels_tech=labels_tech,
            labels_vuln=labels_vuln,
            label_manip=label_manip
        )
        
        loss = outputs["loss"]
        
        return (loss, outputs) if return_outputs else loss


def create_training_args(config: Dict[str, Any]) -> TrainingArguments:

    os.makedirs(config['output']['model_dir'], exist_ok=True)
    os.makedirs(config['output']['logs_dir'], exist_ok=True)
    
    args = TrainingArguments(
        output_dir=config['output']['model_dir'],

        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        num_train_epochs=config['training']['num_epochs'],
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        lr_scheduler_type=config['training']['lr_scheduler'],

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_dir=config['output']['logs_dir'],
        logging_steps=10,
        report_to="tensorboard",

        seed=config['data']['seed'],
        fp16=True
    )
    
    return args


def train_model(
    model: MentalManipMultiTask,
    train_dataset,
    eval_dataset,
    tokenizer,
    config: Dict[str, Any]
) -> MultiTaskTrainer:

    args = create_training_args(config)

    callbacks = []
    if config['training']['early_stopping_patience'] > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config['training']['early_stopping_patience']
            )
        )

    trainer = MultiTaskTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks
    )

    print("Starting training...")
    
    trainer.train()

    final_path = os.path.join(config['output']['model_dir'], "final_model.pt")
    model.save(final_path)
    
    return trainer
