# Module 1 — Data Pipeline

## Project Overview

This module implements a complete data pipeline for book catalog data from Books to Scrape.

The pipeline follows these steps:

Scrape → Clean → Convert → Store in SQLite → Query with SQL → Verify with pandas

---

## Data Source

The data was collected from:

https://books.toscrape.com/

Books to Scrape is a public website created specifically for practicing web scraping.

No login, API key, or paid service is required.

The scraper collects books from multiple categories and pagination pages.

The final dataset contains at least 60 books across at least 3 categories.

---

## Technologies Used

- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite
- sqlite3

---

## Files

### scraper.py

Scrapes the book data and performs data cleaning.

It extracts:

- title
- price
- star rating
- availability
- category

It then converts the fields into the required data types.

### database.py

Creates the normalized SQLite database.

The database contains two related tables:

- categories
- books

The `category_id` column is the foreign key connecting the two tables.

### sql_queries.py

Executes SQL queries demonstrating:

- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- IN
- BETWEEN
- JOIN

The query strings and outputs are saved in `query_outputs.txt`.

### pandas_verification.py

Reads SQL query results using `pd.read_sql()` and reproduces the JOIN using `pd.merge()`.

The results are compared to verify that both approaches produce equivalent results.

### books_cleaned.csv

Contains the cleaned and converted book data.

### books_database.db

SQLite database containing the normalized tables.

### query_outputs.txt

Contains the SQL queries and their outputs.

### pandas_verification.txt

Contains the pandas verification results and the SQL JOIN vs `pd.merge()` comparison.

### requirements.txt

Contains the Python packages required to run this module.

---

## Data Cleaning Decisions

### Price

The currency symbol was removed from the scraped price.

The resulting value was converted to a floating-point number and stored as:

`price_gbp`

### Star Rating

The text ratings were converted to integers:

- One → 1
- Two → 2
- Three → 3
- Four → 4
- Five → 5

### Availability

The availability text was converted into a Boolean column:

- `True` → book is in stock
- `False` → book is not in stock

### Missing Numeric Values

If an unexpected parsing error produces a missing numeric value, the median of the available values is used for imputation.

This prevents the pipeline from crashing because of an unexpected scraped value.

Rows with missing essential fields such as title or category are dropped because these fields are required to identify and categorize a book.

---

## Currency Conversion

The project-required fixed conversion rate was used:

**1 GBP = 105.50 INR**

The `price_inr` column is calculated as:

`price_inr = price_gbp × 105.50`

This is the fixed project-defined baseline and does not require an external currency API.

---

## Database Design

The SQLite database uses a normalized two-table design.

### categories

- `category_id` — Primary Key
- `category_name` — Unique category name

### books

- `book_id` — Primary Key
- `title`
- `price_gbp`
- `price_inr`
- `rating`
- `in_stock`
- `category_id` — Foreign Key referencing `categories.category_id`

This design avoids repeatedly storing the category name for every book and establishes a proper primary-key/foreign-key relationship.

---

## Running the Pipeline

Install the required packages:

```bash
pip install -r requirements.txt