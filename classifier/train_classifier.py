from dataloader import get_dataset
import random
import torch
import datasets

def shuffle_together(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return list(a), list(b)

sentences, labels = get_dataset()
sentences, labels = shuffle_together(sentences, labels)
labels = torch.tensor(labels, device = 'cuda').long()
n_data = len(sentences)
eval_frac = 0.8
eval_partition = int(n_data * eval_frac)

train_sentences, train_labels = sentences[:eval_partition], labels[:eval_partition]
eval_sentences, eval_labels = sentences[eval_partition:], labels[eval_partition:]

raw_datasets = datasets.DatasetDict()
raw_datasets["train"] = datasets.Dataset.from_dict(
        {'text' : train_sentences, 'label' : train_labels})
raw_datasets["test"] = datasets.Dataset.from_dict(
        {'text' : eval_sentences, 'label' : eval_labels})

from transformers import AutoTokenizer, \
        AutoModelForSequenceClassification, \
        TrainingArguments, \
        Trainer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(samples):
    return tokenizer(samples['text'], padding = 'max_length', truncation = True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched = True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels = 2)
training_args = TrainingArguments("test_trainer")

trainer = Trainer(model = model, args = training_args,
        train_dataset = small_train_dataset, eval_dataset = small_eval_dataset)
trainer.train()
