# Scribophile Scraper

## Environment Setup

Here are the steps I took to create my local environment.

- `conda create -n scribophile python=3.7`
- `pip install selenium`
- `pip install webdriver-manager`
- `pip install postgres`
	- Make sure postgres itself is installed on your machine
- `pip install psycopg2`

## Database Setup

The script assumes you have a local postgres database already set up. This is important as it's what supports resuming the scraper in a smart manner should the scraper fail part way through. Feel free to change the user, password, or database name as the database is currently for local execution only.

- `psql -h 127.0.0.1 -p 5432 -U postgres` 
- inside psql shell: `CREATE DATABASE scraper;`
- Setup `scraper/database.ini` config file as per the template below.

```ini
[postgresql]
host=127.0.0.1
database=scraper
user=postgres
password=<your password>

```

# Scraper Stages

1. **Initialization:**
	- Setup thread pool
	- Initialize selenium web driver
	- Set up tables to store metadata, if they don't already exist
2. **Top-Level Story Metadata:** 
	- The scraper's initial pass gathers what metadata it can (e.g. story title or author) from all of the paginated stories available from [Critique Circle's older story page](https://www.critiquecircle.com/queue.asp?status=3). Given how Selenium operates and the restriction of how Critique Circle is designed, we have to gather all the metadata first *before* moving on to individual story pages. Specifically, if we leave the older story page, then we lose our place within the paginated tables.
	- Once the metadata is gathered, it is then saved to postgres. The script can be configured to skip the initial metadata step should later processing fail on something.
		- I could not think of a way to resume the initial metadata scraping should that fail. The problem I ran into was how to resume navigating the paginated tables should we stop part way through.
3. **Scraping Actual Story and Critiques:**
	- Stub