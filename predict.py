# Applied Machine Learning: Kaggle Challenge
# 18/12/2022
# Group 12: Conor O'Donell, Jasper Koppen, Tim Schouten and Mayla Kersten
#
# File contains code to perform predictions for Kaggle, either for an individual model or ensembling. 
# Default is an individual model, uncomment line 100 for ensembling.
#
# Code for predictions based on Inference section from https://huggingface.co/docs/transformers/tasks/multiple_choice
# Confusion matrion matrix source is http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

import pandas as pd
import torch
from transformers import AlbertTokenizer, AlbertForMultipleChoice, AutoTokenizer, AutoModelForMultipleChoice, RobertaTokenizer, RobertaForMultipleChoice
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModelForMultipleChoice.from_pretrained(f"./models/results/bert-base-uncased/checkpoint-5000/")

albert_tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
albert_model = AlbertForMultipleChoice.from_pretrained(f"./models/results/albert-base-v2/checkpoint-3500")

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaForMultipleChoice.from_pretrained(f"./models/results/roberta-base/checkpoint-1000")

options = ["OptionA", "OptionB", "OptionC"]
answers = ['A', 'B', 'C']

MODEL = roberta_model
TOKENIZER = roberta_tokenizer
WEIGHTS = [0.3, 0.3, 0.4]

def predict(example, model_name="bert"):
    """
        Predict option A, B or C by using an individual model.

        Args:
            example = row in the dataframe.
            model_name = name of the model to use, default is "bert"".

        Returns: 
            Predicted label from answers.
    """
    prompts = []
    for opt in options:
        prompts.append([example['FalseSent'], example[opt]])

    tokenizer = eval(model_name + '_tokenizer')
    model = eval(model_name + '_model')

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    labels = torch.tensor(0).unsqueeze(0)
    
    outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)

    predicted_class = outputs.logits.argmax().item()

    return answers[predicted_class]

def ensemble(example, weights=False):
    """
        Prediction option A, B or C by using ensembling method.

        Args:
            example = row in the dataframe.
            weights = optional argument, for a weighted average.

        Returns: 
            Predicted label from answers.
    """
    prompts = []
    for opt in options:
        prompts.append([example['FalseSent'], example[opt]])

    bert_inputs = bert_tokenizer(prompts, return_tensors="pt", padding=True)
    albert_inputs = albert_tokenizer(prompts, return_tensors="pt", padding=True)
    roberta_inputs = roberta_tokenizer(prompts, return_tensors="pt", padding=True)

    labels = torch.tensor(0).unsqueeze(0)
    
    bert_outputs = bert_model(**{k: v.unsqueeze(0) for k, v in bert_inputs.items()}, labels=labels)
    albert_outputs = albert_model(**{k: v.unsqueeze(0) for k, v in albert_inputs.items()}, labels=labels)
    roberta_outputs = roberta_model(**{k: v.unsqueeze(0) for k, v in roberta_inputs.items()}, labels=labels)

    if weights:
        logits = ((bert_outputs.logits * WEIGHTS[0]) + (albert_outputs.logits * WEIGHTS[1]) + (roberta_outputs.logits * WEIGHTS[2]))
    else:
        logits = (bert_outputs.logits + albert_outputs.logits + roberta_outputs.logits) / 3

    predicted_class = logits.argmax().item()

    return answers[predicted_class]

def convert(label):
    """
        Reformats label to correspond with columns.    
    """
    if label == 'A':
        return answers[label]
    elif label == 'B':
        return answers[label]
    else:
        return answers[label]

def analyze(test_file, ensemble=True):
    """
        Analyzes the actual and predicted labels.

        Args:
            test_file = name of file, result from train.py load_data if write_out is set on true
            ensemble = optional argument, in case the analysis is to be done on ensemble.
    """
    df = pd.read_csv(test_file)

    df["predicted"] = df.apply(lambda row: predict(row), axis=1)
    df["actual"] = df["label"].apply(lambda row: convert(row))

    cm = confusion_matrix(df["actual"], df["predicted"], labels=answers)
    
    plot_cm(cm, answers, title='Confusion matrix')

    if ensemble:
        models = ['bert', 'albert', 'roberta']

        # Get predictions per model
        for model in models:
            df[model] = df.apply(lambda row: predict(row, model.split('-')[0]), axis=1)

        # Show Results 
        with open("disagreements.txt", "w") as f:
            for i in range(len(df)):
                example = df.loc[i]

                if not (example[models[0]] == example[models[1]] == example[models[2]]):
                    f.write(write(df.loc[i]), models)

def plot_cm(cm, labels, title=""):
    """
        Plots confusion matrix and saves it.

        Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if labels:
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, 1 - accuracy))
    plt.savefig('confusion_matrix.png')

def write(example, models):
    """
        Writes example in the dataset, used for analysis of ensemble method.

        Args:
            example = item in dataset.
            models = list of models that need to be written out. 
    """
    content = f"FalseSent: {example['FalseSent']}\n"
    content += f"\nA) {example['OptionA']}; B) {example['OptionB']}; C) {example['OptionC']}\n"

    gold_label = example['label']
    content += f"True label: {options[gold_label]}\n"

    mod = ""
    for i in models:
        mod += f"{i}: {example[i]}\t"
    
    content += f"{mod}\n"
    content += "----------------------------------------------\n"

    return content

if __name__ == "__main__":
    test_file = "common-sense/test_data.csv"
    df_test = pd.read_csv(test_file)

    df_test['label'] = df_test.apply(lambda row: predict(row), axis=1)
    df_test['label'] = df_test.apply(lambda row: ensemble(row), axis=1)

    submission_df = df_test[["id", "label"]].copy()
    submission_df = submission_df.rename(columns={"label": "answer"})
    submission_df.set_index("id", inplace=True)
    submission_df.to_csv(f"submission.csv")

    analyze("testset/30_percent_trainset_full.csv")