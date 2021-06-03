from postgres import Postgres
from configparser import ConfigParser
import psycopg2

def config(filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
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
            submission_id bigserial PRIMARY KEY, \
            author varchar, \
            author_link varchar, \
            story_title varchar, \
            story_link varchar, \
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

def insert_story_metadata(table_name, metadata):
    try:
        params = config()
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        psql = conn.cursor()

        query_string = "INSERT INTO " + table_name + " VALUES "
        count = 1
        max_count = len(metadata)
        #for m in metadata:
        m = metadata[0]
        query_string += "(" + \
            m.story_id + ", " + \
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
        #    if count == max_count:
        #        query_string += ";"
        #    else:
        #        query_string += ", "
        #    count += 1

        print(query_string)

        psql.execute(query_string)
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            psql.close()
            print('Database connection closed.')