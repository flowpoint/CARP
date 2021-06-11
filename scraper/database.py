from postgres import Postgres
from configparser import ConfigParser
from scraper_classes import *
import psycopg2

def config(filename='database.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db

def run_connection_health_check():
    try:
        params = config()
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        psql = conn.cursor()

        psql.execute('SELECT version()')
        db_version = psql.fetchone()
        print(db_version)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')

def create_story_metadata_table(group_name):
    try:
        params = config()
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        psql = conn.cursor()

        psql.execute('CREATE TABLE IF NOT EXISTS ' + group_name + ' ( \
            submission_id SERIAL UNIQUE PRIMARY KEY, \
            story_title varchar, \
            story_link varchar UNIQUE NOT NULL, \
            author varchar, \
            author_link varchar, \
            word_count varchar, \
            genre varchar, \
            crit_count varchar, \
            author_notes varchar, \
            story_text varchar, \
            is_processed boolean \
        )')
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            psql.close()
            print('Database connection closed.')

def create_critique_table():
    try:
        params = config()
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        psql = conn.cursor()

        psql.execute('CREATE TABLE IF NOT EXISTS story_critiques ( \
            critique_id SERIAL UNIQUE PRIMARY KEY, \
            submission_id integer, \
            critic_name varchar, \
            critic_link varchar, \
            critique_link varchar UNIQUE NOT NULL, \
            word_count varchar, \
            critique_type varchar, \
            story_target varchar, \
            target_comment varchar \
        )')
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            psql.close()
            print('Database connection closed.')            

def insert_story_metadata(table_name, metadata):
    try:
        params = config()
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        psql = conn.cursor()

        query_string = "INSERT INTO " + table_name + \
            " (author, author_link, story_title, story_link, word_count, genre," + \
            " crit_count, author_notes, story_text, is_processed) VALUES "
        count = 1
        max_count = len(metadata)
        for m in metadata:
            #m = metadata[0]
            query_string += "(" + \
                "$$" + str(m.story_title) + "$$, " + \
                "$$" + str(m.story_link) + "$$, " + \
                "$$" + str(m.author) + "$$, " + \
                "$$" + str(m.author_link) + "$$, " + \
                "$$" + str(m.word_count) + "$$, " + \
                "$$" + str(m.genre) + "$$, " + \
                "$$" + str(m.crit_count) + "$$, " + \
                "$$" + str(m.author_notes) + "$$, " + \
                "$$" + str(m.story_text) + "$$, " + \
                "FALSE" + \
            ")"
            if count == max_count:
                query_string += " ON CONFLICT DO NOTHING ;"
            else:
                query_string += ", "
            count += 1

        psql.execute(query_string)
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            psql.close()
            print('Database connection closed.')

def get_stored_critiques():
    try:
        params = config()
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        psql = conn.cursor()

        query_string = "SELECT * FROM story_critiques;"

        psql.execute(query_string)
        records = psql.fetchall()
        conn.commit()

        critiques = []
        for row in records:
            sc = StoredCritique(
                row[0], # critique_id
                row[1], # submission_id
                row[2], # critic_name
                row[3], # critic_link
                row[4], # critique_link
                row[5], # word_count
                row[6], # critique_type
                row[7], # story_target
                row[8], # target_comment
            )
            critiques.append(sc)

        return critiques
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            psql.close()
            print('Database connection closed.')

def get_stored_metadata(table_name, status="false"):
    try:
        params = config()
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        psql = conn.cursor()

        query_string = "SELECT * FROM " + table_name + \
            " WHERE is_processed=" + status + ";"

        psql.execute(query_string)
        records = psql.fetchall()
        conn.commit()

        metadata = []
        for row in records:
            sm = StoredMetadata(
                row[0], # story_id
                row[1], # story_title
                row[2], # story_link
                row[3], # author
                row[4], # author_link
                row[5], # word_count
                row[6], # genre
                row[7], # crit_count
                row[8], # author_notes
                row[9], # story_text
                row[10], # is_processed
            )
            metadata.append(sm)

        return metadata
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            psql.close()
            print('Database connection closed.')

def update_metadata(table_name, submission_id):
    try:
        params = config()
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        psql = conn.cursor()

        query_string = "UPDATE " + table_name + " " + \
            "SET is_processed = true WHERE submission_id = " + str(submission_id) + ";"

        psql.execute(query_string)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            psql.close()
            print('Database connection closed.')

def insert_critiques(critiques):
    if len(critiques) == 0:
        return

    try:
        params = config()
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        psql = conn.cursor()

        query_string = "INSERT INTO story_critiques " + \
            " (submission_id, critic_name, critic_link, critique_link, word_count, critique_type, story_target, target_comment) VALUES "

        count = 1
        max_count = len(critiques)
        for c in critiques:
            query_string += "(" + \
                "$$" + str(c.submission_id) + "$$, " + \
                "$$" + str(c.critic_name) + "$$, " + \
                "$$" + str(c.critic_link) + "$$, " + \
                "$$" + str(c.critique_link) + "$$, " + \
                "$$" + str(c.word_count) + "$$, " + \
                "$$" + str(c.critique_type) + "$$, " + \
                "$$" + str(c.story_target) + "$$, " + \
                "$$" + str(c.target_comment) + "$$ " + \
            ")"
            if count == max_count:
                query_string += " ON CONFLICT DO NOTHING ;"
            else:
                query_string += ", "
            count += 1

        psql.execute(query_string)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            psql.close()
            print('Database connection closed.')