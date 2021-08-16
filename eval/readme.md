# Evaluation Methods: 
These eval scripts are setup to work on the pytorch model, and expect a pytorch checkpoint with generic name "checkpoint.pt" to be present in the directory.

1. Randomly sample failure cases (test\_fails.py): Select a batch of given size from either training or validation set and show passages where it failed to select the correct review. From here, shows all reviews and assigned confidences to those reviews respective to the passage that failed. Expects our dataset, though you could modify dataloading.py and insert your own dataset if you wanted to.
  
2. Create examples (create\_pairs.py): Manually type passages followed by reviews (differentiates by line, stories shouldn't have actual newlines). Where these typed passage review pairs are the batch, does the same thing the above script does.

3. Compare passages and reviews (check_contrastive_scores.py): Manually type passages and reviews (similar formatting as above, but all passages typed then all reviews). Prints scores between all passages and reviews.
