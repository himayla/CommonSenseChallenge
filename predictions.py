# Predictions for ensembling
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import torch

CHECKPOINT = "checkpoint-1000"
BERT = "bert-base-uncased"

bert_tokenizer = AutoTokenizer.from_pretrained(BERT)
bert_model = AutoModelForMultipleChoice.from_pretrained(f"./models/results/{BERT}/{CHECKPOINT}/")

options = ["OptionA", "OptionB", "OptionC"]
answers = ['A', 'B', 'C']

def ensemble(example):
    prompts = []
    for opt in options:
        prompts.append([example['FalseSent'], example[opt]])

    bert_inputs = bert_tokenizer(prompts, return_tensors="pt", padding=True)

    labels = torch.tensor(0).unsqueeze(0)
    
    bert_outputs = bert_model(**{k: v.unsqueeze(0) for k, v in bert_inputs.items()}, labels=labels)

    predicted_class = bert_outputs.logits.argmax().item()

    return answers[predicted_class]

if __name__ == "__main__":
    test_file = "common-sense/test_data.csv"
    df_test = pd.read_csv(test_file)

    df_test['label'] = df_test.apply(lambda row: ensemble(row), axis=1)

    submission_df = df_test[["id", "label"]].copy()
    submission_df = submission_df.rename(columns={"label": "answer"})
    submission_df.set_index("id", inplace=True)
    submission_df.to_csv(f"submissions/Submission_12-12-2022_1.csv")