from selenium import webdriver
from time import sleep
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException


from database import *

import multiprocessing as mp

import threading

import csv

threadLocal = threading.local()

def get_driver():
    driver = getattr(threadLocal, 'driver', None)
    if driver is None:
        chromeOptions = webdriver.ChromeOptions()
        #chrome_options.add_argument("--disable-extensions")
        #chrome_options.add_argument("--disable-gpu")
        #chrome_options.add_argument("--no-sandbox") # linux only
        chromeOptions.add_argument("--headless")
        driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=chromeOptions)
        setattr(threadLocal, 'driver', driver)
    
    return driver



cc_up_for_review_url = "https://www.critiquecircle.com/queue.asp?status=1"
cc_upcoming_url = "https://www.critiquecircle.com/queue.asp?status=2"
cc_older_stories_url = "https://www.critiquecircle.com/queue.asp?status=3"
cc_options_url = "https://www.critiquecircle.com/queue.asp?action=options"

# NOTE(JR): Given the naming scheme, I have to assume these ids are
# extremely liable to be changed. This should be considered fragile
# These are used to ID tables in the old stories page
newbie_metadata_table = "newbie_queue_metadata"
newbie_table_id = "queue_349"
newbie_block_id = "qd349"

general_metadata_table = "general_metadata"
general_table_id = "queue_1"
general_block_id = "qd1"

fantasy_metadata_table = "fantasy_metadata"
fantasy_table_id = "queue_7"
fantasy_block_id = "qd7"

scifi_metadata_table = "scifi_metadata"
scifi_table_id = "queue_1247"
scifi_block_id = "qd1247"

romance_metadata_table = "romance_metadata"
romance_table_id = "queue_8"
romance_block_id = "qd8"

ya_metadata_table = "ya_metadata"
ya_table_id = "queue_26"
ya_block_id = "qd26"

suspense_metadata_table = "suspense_metadata"
suspense_table_id = "queue_540"
suspense_block_id = "qd540"

# CSV files do *not* like having line breaks in their stored strings
# A standard workaround is to replace "\n" with an uncommonly used symbol
# I picked the Ϯ (coptic capital letter dei) as it seemed unlikely to
# appear in our data set
line_break_replacement = "\u03EE"


class StoryMetadata:
    def __init__(self, story_title, story_link, author, author_link, word_count, genre, crit_count):
        self.story_title = story_title
        self.story_link = story_link
        self.author = author
        self.author_link = author_link
        self.word_count = word_count
        self.genre = genre
        self.crit_count = crit_count

class CritiqueMetadata:
    def __init__(self, story_id, critic_name, critic_link, critique_link, word_count, critique_type):
        self.story_id = story_id
        self.critic_name = critic_name
        self.critic_link = critic_link
        self.critique_link = critique_link
        self.word_count = word_count
        self.critique_type = critique_type

def login():
    # IMPORTANT(JR): please enter your own cc email and password
    # moreover, please do not commit your user/pass to the repo
    user = 'jrsmith17@protonmail.com'
    password = 'Vhnv5o@ExieE8*tqxxtIudJw7fjU'

    driver = get_driver()

    driver.get('https://new.critiquecircle.com/login')

    username_box = driver.find_element_by_xpath("//input[@type='username']")
    username_box.send_keys(user)

    password_box = driver.find_element_by_xpath("//input[@type='password']")
    password_box.send_keys(password)

    login_box = driver.find_element_by_xpath("//button[@type='submit']")
    login_box.click()


def get_next_button(block_id):
    driver = get_driver()
    return driver.find_element_by_css_selector("#" + block_id + " table.FaintBorderBlue td.smalltext:nth-of-type(2) a:nth-last-child(2)")


def gather_per_table_worker(metadata, row):
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


def gather_metadata_callback(result):
    print("call back for gather metadata:",result)

def gather_per_table(table_id, block_id, next_button, pagination_count):
    driver = get_driver()
    metadata = list()

    # IMPORTANT(JR): use the pagination_count for dev purposes only
    # uncomment the second loop to capture the full data set
    gather_count = 0;
    #while next_button:
    while next_button and pagination_count <= 2:
        print("Gathering on",table_id,"Count:",gather_count)
        gather_count += 1
        table = driver.find_element_by_css_selector("#" + table_id)

        pool = mp.Pool()
        for row in table.find_elements_by_css_selector("tr.or"):
            pool.apply_async(gather_per_table_worker, args=(metadata, row), callback=gather_metadata_callback)
        pool.close()
        pool.join()

        # preparation for next loop iteration
        if next_button:
            next_button.click()
            next_button = get_next_button(block_id)
            if next_button.text != ">>":
                break
            pagination_count += 1
        else:
            break

    return metadata

def gather_metadata():
    # Need to ensure type metadata appears in results table
    driver = get_driver()
    driver.get(cc_options_url)
    type_checkbox = driver.find_element_by_css_selector("input#Type")
    if type_checkbox.is_selected() is False:
        type_checkbox.click()

    # Load up main queue page
    driver.get(cc_older_stories_url)
    metadata = list()

    # ########################################
    # Newbie Queue
    # ########################################
    newbie_next_button = get_next_button(newbie_block_id)
    newbie_pagination_count = 0

    newbie_metadata = gather_per_table(newbie_table_id, newbie_block_id, newbie_next_button, newbie_pagination_count)
    metadata.extend(newbie_metadata)

    # IMPORTANT(JR): uncomment this section for a proper run, I left commented out so
    # that the person using this script could test on their machine quickly before
    # committing to a full run
    
    """
    # ########################################
    # General Queue
    # ########################################
    general_next_button = get_next_button(general_block_id)
    general_pagination_count = 0

    general_metadata = gather_per_table(general_table_id, general_block_id, general_next_button, general_pagination_count)
    metadata.extend(general_metadata)

    # ########################################
    # Fantasy Queue
    # ########################################
    fantasy_next_button = get_next_button(fantasy_block_id)
    fantasy_pagination_count = 0

    fantasy_metadata = gather_per_table(fantasy_table_id, fantasy_block_id, fantasy_next_button, fantasy_pagination_count)
    metadata.extend(fantasy_metadata)

    # ########################################
    # Scifi Queue
    # ########################################
    scifi_next_button = get_next_button(scifi_block_id)
    scifi_pagination_count = 0

    scifi_metadata = gather_per_table(scifi_table_id, scifi_block_id, scifi_next_button, scifi_pagination_count)
    metadata.extend(scifi_metadata)

    # ########################################
    # Romance Queue
    # ########################################
    romance_next_button = get_next_button(romance_block_id)
    romance_pagination_count = 0

    romance_metadata = gather_per_table(romance_table_id, romance_block_id, romance_next_button, romance_pagination_count)
    metadata.extend(romance_metadata)

    # ########################################
    # YA Queue
    # ########################################
    ya_next_button = get_next_button(ya_block_id)
    ya_pagination_count = 0

    ya_metadata = gather_per_table(ya_table_id, ya_block_id, ya_next_button, ya_pagination_count)
    metadata.extend(ya_metadata)

    # ########################################
    # Suspense Queue
    # ########################################
    suspense_next_button = get_next_button(suspense_block_id)
    suspense_pagination_count = 0

    suspense_metadata = gather_per_table(suspense_table_id, suspense_block_id, suspense_next_button, suspense_pagination_count)
    metadata.extend(suspense_metadata)
    """

    return metadata

def save_metadata(metadata):
    
    print("saving")

def process_metadata(metadata):
    submission_csv_headers = ['submission_id', 'author', 'author_link', 'story_title',
                              'story_link', 'word_count', 'genre', 'crit_count', 'author_notes', 'story_text']
    

    submission_csv = open('critiquecircle_submissions.csv', 'w', newline='')
    sw = csv.writer(submission_csv)
    sw.writerow(submission_csv_headers) 

    critiques = list()

    try:
        driver = get_driver()
        metadata_count = 0
        for m in metadata:
            print("Processing metadata:",metadata_count)
            metadata_count += 1
            # ###############################################
            # Main Story Submission File
            # ###############################################
            driver.get(m.story_link)

            # The forbidden icon appears on 18+ pages, which we're skipping
            forbidden_icon = driver.find_elements_by_css_selector('img[src*="images/forbidden"]')
            if len(forbidden_icon) > 0:
                continue

            author_notes = ""
            try:
                author_notes = driver.find_element_by_css_selector(".authornotes").text \
                    .replace("\n", line_break_replacement).encode('utf-8')
            except NoSuchElementException:
                author_notes = ""
            story_chunks = driver.find_elements_by_css_selector("#story p")
            story_text = ""
            for sc in story_chunks:
                story_text += sc.text.strip()
                story_text += "\n"
            story_text = story_text.replace("\n", line_break_replacement).encode('utf-8')

            story_id = str(hash(m.author + m.story_title))

            sw.writerow([
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
            ])

            # ###############################################
            # Story Critique File
            # ###############################################
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

                critique_id = 0

                cm = CritiqueMetadata(
                    story_id, 
                    critic_name, 
                    critic_link, 
                    critique_link, 
                    word_count, 
                    critique_type
                )
                critiques.append(cm)
    finally:
        submission_csv.close()
        
    return critiques


def process_critiques(critiques):
    # critic_link, critique_link, word_count, critique_type
    critique_csv_headers = ['comment_id', 'submission_id', 'critic_name', 'critic_link', 'critique_link', 'word_count', 'critique_type', 'story_target', 'target_comment']
    critique_csv = open('critiquecircle_critiques.csv', 'w', newline='')
    cw = csv.writer(critique_csv)
    cw.writerow(critique_csv_headers)

    try:
        driver = get_driver()
        critique_count = 0
        for critique in critiques:
            print("Processing critique:",critique_count)
            critique_count += 1
            driver.get(critique.critique_link)

            comment_count = 0
            comments = driver.find_elements_by_css_selector('div[id^="c_"')
            for c in comments:
                comment_text = c.text.replace("\n", line_break_replacement).encode('utf-8')
                comment_num = c.get_attribute('id').replace("c_", "")
                comment_target = driver.find_element_by_xpath("//p[@onclick=\"ToggleComment(this, " + comment_num + ");\"]")
                target_text = comment_target.text.replace("\n", line_break_replacement).encode('utf-8')

                comment_id = str( hash(critique.story_id + critique.critic_name + str(comment_count) ) )

                cw.writerow([
                    comment_id,
                    critique.story_id,
                    critique.critic_name,
                    critique.critic_link,
                    critique.critique_link,
                    critique.word_count,
                    critique.critique_type,
                    target_text,
                    comment_text
                ])
                comment_count += 1
    finally:
        critique_csv.close()    

def main():
    run_connection_health_check()
    create_story_metadata_table(newbie_metadata_table)
    create_story_metadata_table(general_metadata_table)
    create_story_metadata_table(fantasy_metadata_table)
    create_story_metadata_table(scifi_metadata_table)
    create_story_metadata_table(romance_metadata_table)
    create_story_metadata_table(ya_metadata_table)
    create_story_metadata_table(suspense_metadata_table)
    
    login()
    metadata = gather_metadata()
    save_metadata(metadata)
    #critiques = process_metadata(metadata)
    #process_critiques(critiques)


if __name__ == "__main__":
    main()
