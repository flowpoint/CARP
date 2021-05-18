from datasets import load_dataset
import re
import csv
import os

# Skips first OFFSET passages for gathering data
OFFSET = 0

dataset = load_dataset("csv", data_files = 'prompt_responses_full.csv')
dataset = dataset['train']

passages = dataset['response']
reviews = dataset['response_children']

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

def send_to_csv(reviews):
    offset = OFFSET

    # If file already exists, sets offset so you can start continue 
    # from where it left off
    already_exists = os.path.exists("./isreview.csv")
    if already_exists: 
        with open('isreview.csv', 'r', newline = '') as file:
            line = file.readlines()[-1]
            # Find first and second comma to get the PassageID number of last entry
            first_comma = line.find(',')
            second_comma = line[first_comma + 1:].find(',') + first_comma + 1
            offset = int(line[first_comma + 1: second_comma]) # offset for passage id
   
    write_mode = 'w' if not already_exists else 'a'
    with open('isreview.csv', write_mode, newline = '') as file:
        writer = csv.writer(file)
        if not already_exists: writer.writerow(["RowNum", "PassageID", "Comment", "isReview"])
        rowNum = 0
        quit = False
        for passageNum, passageReplies in enumerate(reviews[offset:]):
            for commentNum, comment in enumerate(passageReplies):
                print("(" + str(rowNum) + ", " + str(passageNum + offset) + ") "
                        + comment)
                inp = ensureInput()
                if inp == -1:
                    quit = True
                    break
                writer.writerow([rowNum, passageNum + offset, comment, inp])
                rowNum += 1 
            if quit: break

# Testing if partition is splitting comments correctly
def test_partition(reviews, start_n, end_n):
    for i, passage in enumerate(reviews[start_n:first_n]):
        for j, review in enumerate(passage):
            print(str(i) + ", " + str(j) + " | " + review)
            
from data_util import partition_review, filter_empty

filter_empty(passages, reviews)
reviews = [partition_review(rev) for rev in reviews]

send_to_csv(reviews)
