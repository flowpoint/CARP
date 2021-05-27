#from bs4 import BeautifulSoup;
#import time

#import urllib.request;

from selenium import webdriver
from time import sleep
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

driver = webdriver.Chrome(ChromeDriverManager().install())

class StoryMetadata:
	def __init__(self, story_link, author_link, critiques_link):
		self.story_link = story_link
		self.author_link = author_link
		self.critiques_link = critiques_link
		#self.word_count = word_count
		#self.story_tags = story_tags


def login():
	user = ''
	password = ''

	
	driver.get('https://www.scribophile.com/dashboard/login')

	username_box = driver.find_element_by_xpath("//input[@type='email']")
	username_box.send_keys(user)

	password_box = driver.find_element_by_xpath("//input[@type='password']")
	password_box.send_keys(password)

	login_box = driver.find_element_by_css_selector("footer button")
	login_box.click()

def gather_metadata():
	driver.get('https://www.scribophile.com/writing/browse?genre=fiction&type=short+story')

	# NOTE(JR): pagination_count is purely for testing so I don't have to sit through it 
	# going through every page during development, this should be removed for a proper run
	next_button = driver.find_element_by_css_selector("a.rel-next")
	pagination_count = 0;
	metadata = list()

	while 'disabled' not in next_button.get_attribute('class').split() and pagination_count <= 2:
		table = driver.find_element_by_xpath("//table[@class='work-list']")

		for row in table.find_elements_by_xpath(".//tr"):
		    work_details = row.find_element_by_css_selector("td.work-details")
		    story_link = work_details.find_elements_by_css_selector("a")[0].get_attribute('href')
		    author_link = work_details.find_elements_by_css_selector("a")[1].get_attribute('href')

		    critiques = row.find_element_by_css_selector("td.critiques")
		    critiques_link = critiques.find_elements_by_css_selector("a")[0]

		    #word_count = row.find_element_by_css_selector("td.words").text
		    # something for story tags, do we need this?

		    sm = StoryMetadata(story_link, author_link, critiques_link)
		    metadata.append(sm)

		# preparation for next loop iteration
		if 'disabled' not in next_button.get_attribute('class').split():
			next_button.click()
			next_button = driver.find_element_by_css_selector("a.rel-next")
			pagination_count += 1
		else:
			break
	
	return metadata

def process_metadata(metadata):
	print("METADATA COUNT:")
	print(len(metadata))

	#for m in metadata:
	#	driver.get(m.story_link)

	driver.get(metadata[0].story_link)
	read_more_button = driver.find_element_by_xpath('//a[@href="#work-body"]')
	read_more_button.click()

def main():
	login()
	metadata = gather_metadata()
	process_metadata(metadata)
	#driver.quit()

if __name__ == "__main__":
    main()