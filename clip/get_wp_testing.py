from datasets import load_dataset
import re
import csv
import os
import torch

# Skips first OFFSET passages for gathering data
OFFSET = 0


# Get input and ensure its 0 or 1 or -1
def ensureInput():
    haveValue = False
    while not haveValue:
        try:
            val = int(input())
            assert (val == 0) or (val == 1) or (val == -1)
            haveValue = True
        except:
            pass
    return val

def remove_invalid(s):
    size = len(s)
    res = ""
    for i in range(size):
        try:
            c = s[i].encode('charmap')
            res += s[i]
        except:
            continue
    return res
            
from data_util import partition_review, filter_empty

# Testing if partition is splitting comments correctly
def test_partition(reviews, start_n, end_n):
    for i, passage in enumerate(reviews[start_n:first_n]):
        for j, review in enumerate(passage):
            print(str(i) + ", " + str(j) + " | " + review)
            
def get_inline(passages, reviews):
    dataset = []
    for i, rev in enumerate(reviews):
        if passages[i] == None or rev == []: continue
        
        rev = partition_review(rev)
        n_rev = len(rev)
        n_spots = len(passages[i])
        
        # pretend reviews are inline
        # generate random spots in text to place them
        pos = torch.randint(n_spots, (n_rev,)).tolist()
        for j in range(n_rev):
            up_to = passages[i][0:pos[j]]
            pair = (up_to, rev[j])
            dataset.append(pair)
    return dataset

def get_dataset():
    dataset = load_dataset("csv", data_files = 'prompt_responses_full.csv')
    dataset = dataset['train']

    passages = dataset['response']
    reviews = dataset['response_children']
    filter_empty(passages, reviews)

    return get_inline(passages, reviews)

