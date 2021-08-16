from dataloading import get_dataset
from testing_util import get_logit

print("Dataset (0) or validation set (1)?")
choice = int(input())

if choice == 0:
    dataset, _ = get_dataset()
else:
    _, dataset = get_dataset()

while True:
    print("Contrastive batch size?")
    batch = int(input())
    if batch > len(dataset):
        print("Batch size exceeds size of dataset being used")
    get_logit(dataset, batch_size)
