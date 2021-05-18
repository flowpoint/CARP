from constants import *
from transformers import AutoModelForSequenceClassification, \
        AutoTokenizer

# Runs the trained model on all samples
from data_util import partition_review, filter_empty
from datasets import load_dataset

import torch

dataset = load_dataset("csv", data_files = 'prompts_full.csv')
dataset = dataset['train']
passages = dataset['response']
comments = dataset['response_children']

filter_empty(passages, comments)
comments = [partition_review(rev) for rev in comments]

# Will contain tuples with review and confidence in said comment being a review
reviews = []

# Prepare model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# TODO: Add loading checkpoint

with torch.no_grad():
    for i, _ in enumerate(passages):
        review_confs = []
        for j, comment in enumerate(comments[i]):
            comment_tokens = tokenizer.encode_plus(comment, return_tensors = "pt")
            logits = model(**comment_tokens)[0]
            conf_review = logits[0][1]
            review_confs.append(conf_review)
        review_confs = torch.stack(review_confs)
        most_likely = torch.argmax(review_confs).item()
        reviews.append((comments[i][most_likely], review_confs[most_likely].item()))

import csv

# Write it all to a csv file
with open('promptreview.csv', 'w', newline = '') as file:
    writer = csv.writer(file)
    writer.writerow(["Passage", "Review", "ConfidenceInReview"])
    for passage, (review, conf) in zip(passages, reviews):
        writer.writerow([passage, review, conf])

