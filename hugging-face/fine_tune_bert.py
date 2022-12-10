# Convert Reddit data into a format that can be used by the BERT model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# train test split
def read_reddit_split(reddit_csv):
    df = pd.read_csv(reddit_csv)
    texts = df["full text"].tolist()
    labels = df["class"].tolist()
    # 80% train, 10% test, 10% valid
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=1.0 / 8.0, stratify=y_train
    )
    print(f"size of train data is {len(x_train)}")
    print(f"size of test data is {len(x_test)}")
    print(f"size of valid data is {len(x_valid)}")
    return x_train, x_test, x_valid, y_train, y_test, y_valid


x_train, x_test, x_valid, y_train, y_test, y_valid = read_reddit_split(
    "/workspaces/Michelle_Li_NLP_Project/data/reddit_dataset.csv"
)

# tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def tokenize_data(x_train, x_test, x_valid):
    train_encodings = tokenizer(x_train, truncation=True, padding=True)
    test_encodings = tokenizer(x_test, truncation=True, padding=True)
    valid_encodings = tokenizer(x_valid, truncation=True, padding=True)
    return train_encodings, test_encodings, valid_encodings


train_encodings, test_encodings, valid_encodings = tokenize_data(
    x_train, x_test, x_valid
)

# convert to Dataset object
class RedditDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = RedditDataset(train_encodings, y_train)
val_dataset = RedditDataset(valid_encodings, y_valid)
test_dataset = RedditDataset(test_encodings, y_test)

# fine-tune BERT model
training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=10,
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
    compute_metrics=compute_metrics,  # compute metrics
)

trainer.train()

# test model
trainer.evaluate(test_dataset)

# save model
trainer.save_model("./models")
# save tokenizer
tokenizer.save_pretrained("./models/tokenizer")
