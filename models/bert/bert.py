from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PaddingStrategy
from typing import Optional, Union
import numpy as np
import pandas as pd

MODEL_NAME = "bert-base-uncased"
CHECKPOINT = "checkpoint-1000"

options = ['OptionA', 'OptionB', 'OptionC']

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class DataCollatorForMultipleChoice:
    """
        Data collator that will dynamically pad the inputs for multiple choice received.
        Flattens all model inputs, apply padding, unflatten results.
    """
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

def load_data(sentences, labels):
    """
        Load and merge sentences, labels convert to Huggingface DatasetDict
    """
    df_sentences = pd.read_csv(sentences)[:5]
    df_labels = pd.read_csv(labels)[:5]

    # Combine sentences and labels dataframes
    df = pd.merge(df_sentences, df_labels, on="id")

    df = df.rename(columns={"answer": "label"})

    # Encode answer to make it binary
    data = Dataset.from_pandas(df).train_test_split(test_size = 0.3).class_encode_column("label")

    return data

def preprocess_function(examples: Dataset):
    """ Perform preprocessing of the input.
        # Arguments
            Dataset: Huggingface Dataset containing features
        # Output
            Huggingface dataset_dict with the tokenized examples 
        with corresponding input_ids, attention_mask, and labels.
    """
    first = [[i] * 3 for i in examples["FalseSent"]]

    second = [
        [f"{examples[opt][i]}" for opt in options] for i in range(len(examples['FalseSent']))
    ]

    first = sum(first, [])
    sec = sum(second, [])

    # Truncation makes sure to make sure input is not longer than max
    tokenized_examples = tokenizer(first, sec, truncation=True)
 
    return {k: [v[i : i + 3] for i in range(0, len(v), 3)] for k, v in tokenized_examples.items()}

def compute_metrics(eval_predictions):
    """
        Compute metrics from the predictions.
    """
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

if __name__ == "__main__":
    
    sentences = "../../common-sense/train_data.csv"
    labels = "../../common-sense/train_answers.csv"

    data = load_data(sentences, labels)

    tokenized = data.map(preprocess_function, batched=True)

    model = AutoModelForMultipleChoice.from_pretrained(f"{MODEL_NAME}/{CHECKPOINT}")

    training_args = TrainingArguments(
        output_dir = f"{MODEL_NAME}",
        evaluation_strategy = "epoch",
        learning_rate = 5e-5,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        num_train_epochs = 3, # Default = 3
        weight_decay = 0.01,
    )


    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized["train"],
        eval_dataset = tokenized["test"],
        tokenizer = tokenizer,
        data_collator = DataCollatorForMultipleChoice(),
        compute_metrics = compute_metrics,
    )

    trainer.train()