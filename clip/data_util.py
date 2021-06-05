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
