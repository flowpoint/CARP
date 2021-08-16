from testing_util import get_logit

while True:
    pairs = []
    print("Write passages, followed by newline, then review")
    print("Type -1 for passage when finished")
    while True:
        print("PASSAGE:")
        passage = input()
        if passage == "-1": break
        print("REVIEW:")
        review = input()

        pairs.append((passage, review))

    get_logit(pairs, -1)
