# CARP: Contrastive Authoring+Reviewing Pretraining

Using CLIP with two text encoders to learn representations for story passages and reviews corresponding to those passages. The goal of this project is to follow up this pretraining with hand crafted reviews in order to guide a story down a certain path as is being written by the model (controllable natural language generation). By doing this, we hope to ground GPT-NEO.

# Dataset

For this project, we are using a dataset scraped from Critique Circle that consists of stories and inline reviews within those stories.
