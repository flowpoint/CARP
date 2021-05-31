#from bs4 import BeautifulSoup;

from selenium import webdriver
from time import sleep
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

import csv

driver = webdriver.Chrome(ChromeDriverManager().install())


class StoryMetadata:
    def __init__(self, story_link, author_link, critiques_link, word_count, story_tags):
        self.story_link = story_link
        self.author_link = author_link
        self.critiques_link = critiques_link
        self.word_count = word_count
        self.story_tags = story_tags


def login():
    user = 'jrsmith17@protonmail.com'
    password = 'V7Px5y@vduSvj@ysA$JZPDH$J#6CR*o'

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
    pagination_count = 0
    metadata = list()

    while 'disabled' not in next_button.get_attribute('class').split() and pagination_count <= 2:
        table = driver.find_element_by_xpath("//table[@class='work-list']")

        for row in table.find_elements_by_xpath(".//tr"):
            work_details = row.find_element_by_css_selector("td.work-details")
            story_link = work_details.find_elements_by_css_selector("a")[0].get_attribute('href')
            author_link = work_details.find_elements_by_css_selector("a")[1].get_attribute('href')

            critiques = row.find_element_by_css_selector("td.critiques")
            critiques_link = critiques.find_elements_by_css_selector("a")[0].get_attribute('href')

            word_count_raw = row.find_element_by_css_selector("td.words").text
            word_count = int(word_count_raw.replace(" words", "").replace(",", ""))

            story_tags = []
            tags = row.find_elements_by_css_selector("td .tags a")
            for tag in tags:
                story_tags.append(tag.text)

            sm = StoryMetadata(story_link, author_link, critiques_link, word_count, story_tags)
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
    submission_csv_headers = ['id', 'author', 'author_link', 'story_title', 'word_count', 'story_tags', 'about_text', 'story_text']

    critique_csv_headers = ['id', 'submission_id']

    submission_csv = open('scribophile_submissions.csv', 'w')
    sw = csv.writer(submission_csv)
    critique_csv = open('scribophile_critiques.csv', 'w')
    cw = csv.writer(critique_csv)

    # CSV files do *not* like having line breaks in their stored strings
    # A standard workaround is to replace "\n" with an uncommonly used symbol
    # I picked the Ϯ (coptic capital letter dei) as it seemed unlikely to
    # appear in our data set
    line_break_replacement = "\u03EE"

    # for m in metadata:
    #	driver.get(m.story_link)



    try:
        # Main Story Submission File
        sw.writerow(submission_csv_headers)

        driver.get(metadata[0].story_link)
        read_more_button = driver.find_element_by_xpath('//a[@href="#work-body"]')
        read_more_button.click()

        about_raw = driver.find_elements_by_css_selector('#about p')
        about_text = ""
        for paragraph in about_raw:
            about_text += paragraph.text
            about_text += "\n"
        about_text = about_text.replace("\n", line_break_replacement).encode('utf-8')

        story_raw = driver.find_elements_by_css_selector('#work-body p')
        story_text = ""
        for paragraph in story_raw:
            story_text += paragraph.text
            story_text += "\n"
        story_text = story_text.replace("\n", line_break_replacement).encode('utf-8')

        story_title = driver.find_element_by_css_selector('h1').text.encode('utf-8')
        author_name = driver.find_element_by_css_selector('a[rel~="author"]').text.encode('utf-8')
        story_id = str( hash(author_name + story_title) )

        sw.writerow([ \
            story_id, \
            author_name, \
            metadata[0].author_link, \
            story_title, \
            str(metadata[0].word_count), \
            metadata[0].story_tags, \
            about_text, \
            story_text, \
        ])

        # Story Critique File
        # comments aren't always paired and they're not marked with a unique id in any way
        # tabling for now to switch to critique circle
        """
        cw.writerow(critique_csv_headers)
        driver.get(metadata[6].critiques_link)
        critiques = driver.find_elements_by_css_selector('.bubble-list.comments > li')

        for c in critiques:
        	read_more_button = c.find_element_by_css_selector('a.pill-button.down')
        	read_more_button.click()
        """
    finally:
        submission_csv.close()
        critique_csv.close()




def main():
    login()
    metadata = gather_metadata()
    process_metadata(metadata)

    # This is purposefully commented out during dev, during a proper run, please uncomment
    # driver.quit()


if __name__ == "__main__":
    main()
