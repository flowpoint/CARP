# Scribophile Scraper

conda create -n scribophile python=3.7
pip install beautifulsoup4
pip install selenium
pip install webdriver-manager
pip install autopep8


autopep8 main.py --select=E1 -i # THIS IS A BAD TOOL

Running to-do list:
- store both the text w/o comments and the comments itself
	- probably want a separate data file for the raw, full story
- inline comments should also refer to the specific version
- gather all metadata that seems relevant
- keep review colors (useful for categorization)