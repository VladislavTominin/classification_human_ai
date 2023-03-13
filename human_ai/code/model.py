import evaluate
from tqdm import tqdm
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


class RobertaHumanAI:

    def __init__(self, training_args):
        self.model_name = "roberta-base-openai-detector"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)

        self.tokenize_function = tokenize_function
        self.training_args = TrainingArguments(**training_args)
        self.metric = evaluate.load("accuracy")

    def get_tokenized_dataset(self, dataset):
        return dataset.map(self.tokenize_function, batched=True, num_proc=4)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def train(self, dataset):
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['test'],
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
