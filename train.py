# Applied Machine Learning: Kaggle Challenge
# 18/12/2022
# Group 12: Conor O'Donell, Jasper Koppen, Tim Schouten and Mayla Kersten
#
# File contains code fine-tune BERT.
#
# Code based on https://huggingface.co/docs/transformers/tasks/multiple_choice

from datasets import Dataset
import torch
from transformers import AlbertTokenizer, AlbertForMultipleChoice, AutoTokenizer, AutoModelForMultipleChoice, RobertaTokenizer, RobertaForMultipleChoice, TrainingArguments, Trainer
import numpy as np
import pandas as pd

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModelForMultipleChoice.from_pretrained(f"./models/results/bert-base-uncased/checkpoint-5000/")

albert_tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
albert_model = AlbertForMultipleChoice.from_pretrained(f"./models/results/albert-base-v2/checkpoint-3500")

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaForMultipleChoice.from_pretrained(f"./models/results/roberta-base/checkpoint-1000")

MODEL = roberta_model
TOKENIZER = roberta_tokenizer
SEED = 42

class DataCollatorForMultipleChoice:
    """
        Dynamically pad the inputs for multiple choice received.
        Flattens all model inputs, applies padding, and unflatten results.

        Optional parameters to change:
        - padding
        - max_length
        - pad_to_multiple_of
    """
    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = TOKENIZER.pad(
            flattened_features,
            padding=True,
            max_length=None,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

def load_data(sentences, labels, write_out=False):
    """
        Load and merges sentences and labels, split data, encode labels and convert to Huggingface DatasetDict.

        Args:
            sentences = name of file containing the sentences and options.
            labels = name of file containing the correct labels.
            write_out = optional argument, to write out the test data for analytics.

        Returns: 
            DatasetDict containing all train and test data. 
    """
    df_sentences = pd.read_csv(sentences)
    df_labels = pd.read_csv(labels)

    df = pd.merge(df_sentences, df_labels, on="id").rename(columns={"answer": "label"})

    data = Dataset.from_pandas(df).train_test_split(test_size=0.3, seed=SEED).class_encode_column("label")

    if write_out:
        data['test'].to_csv("30_percent_trainset_full.csv", index=False)

    return data

def preprocess_function(examples):
    """ 
        Perform preprocessing of the input.

        Args:
            Dataset: Huggingface Dataset containing features

        Returns:
            DatasetDict with the tokenized examples with corresponding input_ids, attention_mask, and labels.
    """
    options = ['OptionA', 'OptionB', 'OptionC']

    first = [[i] * 3 for i in examples["FalseSent"]]

    second = [
        [f"{examples[opt][i]}" for opt in options] for i in range(len(examples['FalseSent']))
    ]

    first = sum(first, [])
    sec = sum(second, [])

    # Truncation makes sure to make sure input is not longer than max
    tokenized_examples = TOKENIZER(first, sec, truncation=True)
 
    return {k: [v[i : i + 3] for i in range(0, len(v), 3)] for k, v in tokenized_examples.items()}

def compute_metrics(eval_predictions):
    """
        Compute metrics on the test set.
    """
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {
        "accuracy": (preds == label_ids).astype(np.float32).mean().item()
        }

if __name__ == "__main__":
    
    sentences = "common-sense/train_data.csv"
    labels = "common-sense/train_answers.csv"

    data = load_data(sentences, labels)

    tokenized = data.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir = f"./models/results/{MODEL.base_model_prefix}",
        evaluation_strategy = "epoch",
        learning_rate = 5e-5,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        num_train_epochs = 1, # Default = 3
        weight_decay = 0.01,
    )

    trainer = Trainer(
        model = MODEL,
        args = training_args,
        train_dataset = tokenized["train"],
        eval_dataset = tokenized["test"],
        tokenizer = TOKENIZER,
        data_collator = DataCollatorForMultipleChoice(),
        compute_metrics = compute_metrics,
    )

    trainer.train()