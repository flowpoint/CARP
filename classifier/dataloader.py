from csv import reader

# Loading dataset all at once should be fine since its not that large
def get_dataset():
    comments = []
    labels = []
    with open('isreview.csv', 'r', newline = '') as file:
        csv_reader = reader(file)
        for row in csv_reader:
          comments.append(row[2])
          labels.append(row[3])
    return comments, labels
