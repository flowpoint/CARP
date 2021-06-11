from selenium import webdriver
from time import sleep
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

from sys import platform
import sys


from database import *
from scraper_classes import *

import multiprocessing as mp

import threading

import csv

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-sm", "--skip-metadata", action='store_true', help="Skip metadata")


cc_up_for_review_url = "https://www.critiquecircle.com/queue.asp?status=1"
cc_upcoming_url = "https://www.critiquecircle.com/queue.asp?status=2"
cc_older_stories_url = "https://www.critiquecircle.com/queue.asp?status=3"
cc_options_url = "https://www.critiquecircle.com/queue.asp?action=options"

# NOTE(JR): Given the naming scheme, I have to assume these ids are
# extremely liable to be changed. This should be considered fragile
# These are used to ID tables in the old stories page
metadata_tables = [
    "newbie_queue_metadata",
    "general_metadata",
    "fantasy_metadata",
    "scifi_metadata",
    "romance_metadata",
    "ya_metadata",
    "suspense_metadata"
]

table_ids = [
    "queue_349", # newbie queue
    "queue_1", # general
    "queue_7", # fantasy
    "queue_1247", # scifi
    "queue_8", # romance
    "queue_26", # ya
    "queue_540" # suspense
]

block_ids = [
    "qd349", # newbie queue
    "qd1", # general
    "qd7", # fantasy
    "qd1247", # scifi
    "qd8", # romance
    "qd26", # ya
    "qd540" # suspense
]

# CSV files do *not* like having line breaks in their stored strings
# A standard workaround is to replace "\n" with an uncommonly used symbol
# I picked the Ϯ (coptic capital letter dei) as it seemed unlikely to
# appear in our data set
line_break_replacement = "\u03EE"




def login(driver):
    # IMPORTANT(JR): please enter your own cc email and password
    # moreover, please do not commit your user/pass to the repo
    user = ''
    password = ''

    driver.get('https://new.critiquecircle.com/login')

    username_box = driver.find_element_by_xpath("//input[@type='username']")
    username_box.send_keys(user)

    password_box = driver.find_element_by_xpath("//input[@type='password']")
    password_box.send_keys(password)

    login_box = driver.find_element_by_xpath("//button[@type='submit']")
    login_box.click()


def get_next_button(driver, block_id):
    return driver.find_element_by_css_selector("#" + block_id + " table.FaintBorderBlue td.smalltext:nth-of-type(2) a:nth-last-child(2)")

def process_metadata_row(metadata, row):
    # Critique circle allows users to lock content away from accounts that haven't reviewed other stories
    # Assuming that our scraper accounts haven't done any actual reviewing, I figured it best to skip
    locked_icons = row.find_elements_by_css_selector('td:nth-child(1) img[src*="images/shield_"]')
    if len(locked_icons) > 0:
        return "skipped"

    story_title = row.find_element_by_css_selector("td:nth-child(1)").text
    story_link = row.find_element_by_css_selector("td:nth-child(1) a").get_attribute('href')

    author = row.find_element_by_css_selector("td:nth-child(2) a.hoverlink span").text
    author_link = row.find_element_by_css_selector("td:nth-child(2) a.hoverlink").get_attribute('href')

    word_count = int(row.find_element_by_css_selector("td:nth-child(4) nobr").text.replace(",", ""))

    genre = row.find_element_by_css_selector("td:nth-child(5) nobr").text

    crit_count = int(row.find_element_by_css_selector("td:nth-child(6) nobr").text)

    sm = StoryMetadata(story_title, story_link, author, author_link, word_count, genre, crit_count)
    metadata.append(sm)

    return "added"

def gather_metadata(driver, table_name, table_id, block_id):
    driver.get(cc_options_url)
    type_checkbox = driver.find_element_by_css_selector("input#Type")
    if type_checkbox.is_selected() is False:
        type_checkbox.click()

    # Load up main queue page
    driver.get(cc_older_stories_url)
    metadata = list()

    next_button = get_next_button(driver, block_id)
    pagination_count = 0

    while next_button and pagination_count <= 0:
        table = driver.find_element_by_css_selector("#" + table_id)

        for row in table.find_elements_by_css_selector("tr.or"):
            process_metadata_row(metadata, row)

        # preparation for next loop iteration
        if next_button:
            next_button.click()
            next_button = get_next_button(driver, block_id)
            if next_button.text != ">>":
                break
            pagination_count += 1
        else:
            break

    return metadata

def process_row_metadata(driver, metadata):
    stories = list()
    for m in metadata:
        driver.get(m.story_link)

        # The forbidden icon appears on 18+ pages, which we're skipping
        forbidden_icon = driver.find_elements_by_css_selector('img[src*="images/forbidden"]')
        if len(forbidden_icon) > 0:
            continue

        author_notes = ""
        try:
            author_notes = driver.find_element_by_css_selector(".authornotes").text \
                .replace("\n", line_break_replacement)
            #    .replace("\n", line_break_replacement).encode('utf-8')
        except NoSuchElementException:
            author_notes = ""
        story_chunks = driver.find_elements_by_css_selector("#story p")
        story_text = ""
        for sc in story_chunks:
            story_text += sc.text.strip()
            story_text += "\n"
        #story_text = story_text.replace("\n", line_break_replacement).encode('utf-8')
        story_text = story_text.replace("\n", line_break_replacement)

        story_id = str(hash(m.author + m.story_title) + sys.maxsize + 1)
    
        story = FullStoryMetadata(
            story_id,
            m.author,
            m.author_link,
            m.story_title,
            m.story_link,
            m.word_count,
            m.genre,
            m.crit_count,
            author_notes,
            story_text
        )
        stories.append(story)

    return stories

def run_metadata_scraper_manager(table_name, table_id, block_id):
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("--headless")
    if platform == "linux" or platform == "linux2":
        options.add_argument("--no-sandbox")
    
    driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)

    # Per Table Scraping Options
    login(driver)
    row_metadata = gather_metadata(driver, table_name, table_id, block_id)
    full_metadata = process_row_metadata(driver, row_metadata)
    insert_story_metadata(table_name, full_metadata)

    driver.quit()

def run_threaded_metadata_scraper():
    processes = []
    #for i in range(len(metadata_tables)):
    for i in range(1):
        p = mp.Process(target=run_metadata_scraper_manager, args=(metadata_tables[i], table_ids[i], block_ids[i]))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

def run_critique_scraper_manager(table_name):
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("--headless")
    if platform == "linux" or platform == "linux2":
        options.add_argument("--no-sandbox")
    
    driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)
    
    login(driver)
    metadata = get_stored_metadata(table_name)

    critiques = []
    for m in metadata:
        driver.get(m.story_link)

        # update metadata entry to say it's processed
        update_metadata(table_name, m.story_id)

        # The forbidden icon appears on 18+ pages, which we're skipping
        forbidden_icon = driver.find_elements_by_css_selector('img[src*="images/forbidden"]')
        if len(forbidden_icon) > 0:
            continue

        critique_table = driver.find_element_by_css_selector(".tablecls")

        for row in critique_table.find_elements_by_css_selector("tr.or, tr.er"):
            # The HTML used for display the critiques in a table is quite gnarly
            # Specifically, tr.or -or- tr.er could be the last row, which doesn't
            # have 5 columns but instead just one with a link, this bit tests for 
            # the last row and moves on if that's the case
            last_row_text_check = row.find_element_by_css_selector("td:nth-child(1) a").text
            if last_row_text_check == "View all Inline Critiques together":
                continue

            # Doing type first because if it's classic we're skipping
            critique_type = row.find_element_by_css_selector("td:nth-child(5)").text
            if critique_type == "Classic":
                continue

            critique_link = row.find_element_by_css_selector("td:nth-child(1) nobr a").get_attribute('href')
            word_count = row.find_element_by_css_selector("td:nth-child(4)").text

            # The tr/except is necessary here because an Anonymous critique won't have an anchor tag
            # for the scraper to find
            critic_name = ""
            critic_link = ""
            try:
                critic_name = row.find_element_by_css_selector("td:nth-child(2) a").text
                critic_link = row.find_element_by_css_selector("td:nth-child(2) a").get_attribute('href')
            except NoSuchElementException:
                critic_name = 'Anonymous'
                critic_link = 'None'

            cm = CritiqueMetadata(
                m.story_id,
                critic_name, 
                critic_link, 
                critique_link, 
                word_count, 
                critique_type
            )
            critiques.append(cm)

    filled_critiques = []
    for critique in critiques:
            driver.get(critique.critique_link)

            comment_count = 0
            comments = driver.find_elements_by_css_selector('div[id^="c_"')
            for c in comments:
                comment_text = c.text.replace("\n", line_break_replacement)
                comment_num = c.get_attribute('id').replace("c_", "")
                comment_target = driver.find_element_by_xpath("//p[@onclick=\"ToggleComment(this, " + comment_num + ");\"]")
                target_text = comment_target.text.replace("\n", line_break_replacement)

                fc = FullCritique(
                    critique.submission_id,
                    critique.critic_name, 
                    critique.critic_link, 
                    critique.critique_link, 
                    critique.word_count, 
                    critique.critique_type,
                    target_text,
                    comment_text
                )
                filled_critiques.append(fc)
                

    insert_critiques(filled_critiques)

    driver.quit()

def run_threaded_critique_scraper():
    processes = []
    #for i in range(len(metadata_tables)):
    for i in range(1):
        print(metadata_tables[i])
        p = mp.Process(target=run_critique_scraper_manager, args=(metadata_tables[i],) )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

def serialize_stories_to_csv():
    submission_csv_headers = ['submission_id', 'author', 'author_link', 'story_title',
                              'story_link', 'word_count', 'genre', 'crit_count', 'author_notes', 'story_text']
    

    submission_csv = open('critiquecircle_submissions.csv', 'w', newline='')
    sw = csv.writer(submission_csv)
    sw.writerow(submission_csv_headers)

    try:
        for table_name in metadata_tables:
            metadata = get_stored_metadata(table_name, status="true")
            for m in metadata:
                sw.writerow([
                    str(m.story_id).encode('utf-8'),
                    m.author.encode('utf-8'),
                    m.author_link.encode('utf-8'),
                    m.story_title.encode('utf-8'),
                    m.story_link.encode('utf-8'),
                    m.word_count.encode('utf-8'),
                    m.genre.encode('utf-8'),
                    m.crit_count.encode('utf-8'),
                    m.author_notes.encode('utf-8'),
                    m.story_text.encode('utf-8')
                ])
    finally:
        submission_csv.close()

def serialize_critiques_to_csv():
    critique_csv_headers = ['comment_id', 'submission_id', 'critic_name', 'critic_link', 'critique_link', 'word_count', 'critique_type', 'story_target', 'target_comment']
    critique_csv = open('critiquecircle_critiques.csv', 'w', newline='')
    cw = csv.writer(critique_csv)
    cw.writerow(critique_csv_headers)

    try:
        critiques = get_stored_critiques()
        for c in critiques:
            cw.writerow([
                str(c.critique_id).encode('utf-8'),
                str(c.submission_id).encode('utf-8'),
                c.critic_name.encode('utf-8'),
                c.critic_link.encode('utf-8'),
                c.critique_link.encode('utf-8'),
                c.word_count.encode('utf-8'),
                c.critique_type.encode('utf-8'),
                c.story_target.encode('utf-8'),
                c.target_comment.encode('utf-8')
            ])
    finally:
        critique_csv.close()

def main():
    run_connection_health_check()
    for table_name in metadata_tables:
        create_story_metadata_table(table_name)
    create_critique_table()

    args = parser.parse_args()
    if args.skip_metadata is not True:
        run_threaded_metadata_scraper()

    run_threaded_critique_scraper()

    serialize_stories_to_csv()
    serialize_critiques_to_csv()

    ###########
    """
    
    metadata = gather_metadata()
    #critiques = process_metadata(metadata)
    #process_critiques(critiques)
    """

if __name__ == "__main__":
    main()
