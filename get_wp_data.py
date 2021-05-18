from datasets import load_dataset
import re
import csv
import os

# Skips first OFFSET passages for gathering data
OFFSET = 0

dataset = load_dataset("csv", data_files = 'prompts_full.csv')
dataset = dataset['train']

passages = dataset['response']
reviews = dataset['response_children']

# Check if char can be encoded
def check_char(char):
    try:
        x = char.encode('charmap')
        return True
    except:
        return False

def partition_review(rev):
    # They use a single string to store all replies
    # It's like a list of strings (but as a string)
    # A single reply is either encased in \' ... \' or " ... " if the reply contains a \'
    if(rev is None or len(rev) == 2):
        return [] # No reviews

    reviews = []

    match = None
    escape = False

    rev_single = "" # Review to be added to list of reviews
    for char in rev[1:-1]: # iterate with [] removed\
        if match is None: # Starting a new review
            if char == "\"" or char == "\'": # skips comma and space
                match = char
                rev_single = ""
            continue
        elif not escape and match == char: # At the end of a review
            reviews.append(rev_single)
            match = None
        else:
            escape = False
            if char == '\\':
                escape = True
            if check_char(char):
                rev_single += char
    return reviews

# Filter out passages with no reviews
def filter_empty(passages, reviews):
    assert len(reviews) == len(passages)
    
    size = len(passages)
    i = 0
    while i < size:
        if reviews[i] == '[]' or reviews[i] == []:
            del reviews[i]
            del passages[i]
            size -= 1
            continue
        i += 1

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
            
filter_empty(passages, reviews)
reviews = [partition_review(rev) for rev in reviews]

send_to_csv(reviews)
