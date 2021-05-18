from datasets import load_dataset
import re
import csv

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
    rev_single = "" # Review to be added to list of reviews
    for char in rev[1:-1]: # iterate with [] removed\
        if match is None: # Starting a new review
            if char == "\"" or char == "\'": # skips comma and space
                match = char
                rev_single = ""
            continue
        elif char == '\\': # escape character
        elif match == char: # At the end of a review
            reviews.append(rev_single)
            match = None
        else:
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
    with open('isreview.csv', 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(["RowNum", "Passage", "Comment", "isReview"])
        rowNum = 0
        quit = False
        for passageNum, passageReplies in enumerate(reviews):
            for comment in passageReplies:
                print("(" + str(rowNum) + ") " + comment)
                inp = ensureInput()
                if inp == -1:
                    quit = True
                    break
                writer.writerow([rowNum, passageNum, comment, inp])
                rowNum += 1 
            if quit: break

filter_empty(passages, reviews)
print(reviews[9])
exit()
reviews = [partition_review(rev) for rev in reviews]
for i, passage in enumerate(reviews[0:50]):
    for j, review in enumerate(passage):
        print(str(i) + ", " + str(j) + " | " + review)
#filter_empty(passages, reviews)
#send_to_csv(reviews)
