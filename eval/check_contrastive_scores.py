from testing_util import compute_logit

while True:
    passages = []
    reviews = []
    print("Write -1 to specify you're finished")
    print("PASSAGES:")
    while True:
        passage = input()
        if passage == "-1": break
        passages.append(passage)
    print("REVIEWS:")
    while True:
        review = input()
        if review == "-1": break
        reviews.append(review)

    print("SCORES:")
    compute_logit(passages, reviews)


