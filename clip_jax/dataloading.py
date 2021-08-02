from datasets import load_from_disk

from constants import *

def get_dataset():
    dataset = load_from_disk("critiquecircle_critiques_masked_anon")
    train = dataset['train']
    passages = train['story_target']
    reviews = train['target_comment']
    
    res = list(zip(passages, reviews))
    return res[:-VALIDATION_SIZE], res[-VALIDATION_SIZE:]

def get_toy_dataset():
    passages = ["a b c d e f g" for _ in range(2048)]
    reviews = ["h i j k l m n o" for _ in range(2048)]  
    res = list(zip(passages, reviews))
    return res[:-VALIDATION_SIZE], res[-VALIDATION_SIZE:]
