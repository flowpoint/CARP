from datasets import load_dataset
from constants import *

def get_dataset():
    dataset = load_dataset("csv", data_files = 'critique_circle/critiquecircle_critiques.csv')
    dataset = dataset['train']
    passages = dataset['story_target']
    reviews = dataset['target_comment']

    res = list(zip(passages, reviews))
    return res[:-VALIDATION_SIZE], res[-VALIDATION_SIZE:]
