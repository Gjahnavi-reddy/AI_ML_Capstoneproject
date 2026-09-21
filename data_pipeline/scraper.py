import time
import requests
from bs4 import BeautifulSoup
import pandas as pd


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

BASE_URL = "https://books.toscrape.com/"
FIXED_CONVERSION_RATE = 105.50

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# We use 5 categories so that we comfortably exceed
# the minimum requirement of 60 books.

CATEGORIES = {
    "Women's Fiction":
        "https://books.toscrape.com/catalogue/category/books/womens-fiction_9/index.html",

    "Travel":
        "https://books.toscrape.com/catalogue/category/books/travel_2/index.html",

    "Art":
        "https://books.toscrape.com/catalogue/category/books/art_25/index.html",

    "Science":
        "https://books.toscrape.com/catalogue/category/books/science_22/index.html",

    "Fantasy":
        "https://books.toscrape.com/catalogue/category/books/fantasy_19/index.html"
}


# Text ratings → integer ratings

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


# ---------------------------------------------------------
# FUNCTION: GET CATEGORY BOOKS
# ---------------------------------------------------------

def scrape_category(category_name, category_url):
    """
    Scrape all books available in one category.
    """

    books = []

    current_url = category_url
    page_number = 1

    while current_url:

        print(
            f"Scraping {category_name} - page {page_number}"
        )

        try:
            response = requests.get(
                current_url,
                headers=HEADERS,
                timeout=15
            )

            response.raise_for_status()

        except requests.RequestException as error:
            print(
                f"Could not scrape {current_url}: {error}"
            )
            break

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        book_cards = soup.select(
            "article.product_pod"
        )

        if not book_cards:
            print(
                f"No books found on {current_url}"
            )
            break

        # -------------------------------------------------
        # Extract books from current page
        # -------------------------------------------------

        for card in book_cards:

            try:

                # Title
                title_element = card.select_one(
                    "h3 a"
                )

                title = (
                    title_element.get("title", "").strip()
                    if title_element
                    else None
                )

                # Price
                price_element = card.select_one(
                    "p.price_color"
                )

                price_raw = (
                    price_element.get_text(strip=True)
                    if price_element
                    else None
                )

                # Rating
                rating_element = card.select_one(
                    "p.star-rating"
                )

                if rating_element:
                    rating_classes = rating_element.get(
                        "class", []
                    )

                    rating_raw = (
                        rating_classes[1]
                        if len(rating_classes) > 1
                        else None
                    )
                else:
                    rating_raw = None

                # Availability
                availability_element = card.select_one(
                    "p.instock.availability"
                )

                availability_raw = (
                    availability_element.get_text(
                        " ",
                        strip=True
                    )
                    if availability_element
                    else None
                )

                books.append({
                    "title": title,
                    "price_raw": price_raw,
                    "rating_raw": rating_raw,
                    "availability_raw": availability_raw,
                    "category_name": category_name
                })

            except Exception as error:

                # If one book has unexpected HTML,
                # skip that book instead of crashing
                print(
                    f"Skipping one book because of parsing error: {error}"
                )

        # -------------------------------------------------
        # Find next page
        # -------------------------------------------------

        next_button = soup.select_one(
            "li.next a"
        )

        if next_button:

            next_link = next_button.get("href")

            current_url = requests.compat.urljoin(
                current_url,
                next_link
            )

            page_number += 1

            # Small delay between requests
            time.sleep(1)

        else:

            current_url = None

    return books


# ---------------------------------------------------------
# PHASE 1: SCRAPE DATA
# ---------------------------------------------------------

print("=" * 60)
print("PHASE 1: SCRAPING BOOK DATA")
print("=" * 60)

collected_books = []

for category_name, category_url in CATEGORIES.items():

    category_books = scrape_category(
        category_name,
        category_url
    )

    print(
        f"Collected {len(category_books)} books "
        f"from {category_name}"
    )

    collected_books.extend(category_books)

    time.sleep(1)


# Convert scraped records into DataFrame

df_raw = pd.DataFrame(collected_books)

print()
print(
    f"Total raw books collected: {len(df_raw)}"
)

print(
    f"Categories collected: "
    f"{df_raw['category_name'].nunique()}"
)


# ---------------------------------------------------------
# PHASE 2: DATA CLEANING
# ---------------------------------------------------------

print()
print("=" * 60)
print("PHASE 2: CLEANING AND TYPE CONVERSION")
print("=" * 60)

df_clean = df_raw.copy()


# ---------------------------------------------------------
# Clean title
# ---------------------------------------------------------

df_clean["title"] = (
    df_clean["title"]
    .astype("string")
    .str.strip()
)


# ---------------------------------------------------------
# Clean price_gbp
# ---------------------------------------------------------

df_clean["price_gbp"] = pd.to_numeric(
    df_clean["price_raw"]
    .astype("string")
    .str.replace(
        r"[^\d.]",
        "",
        regex=True
    ),
    errors="coerce"
)


# ---------------------------------------------------------
# Clean rating
# ---------------------------------------------------------

df_clean["rating"] = (
    df_clean["rating_raw"]
    .map(RATING_MAP)
)


# ---------------------------------------------------------
# Clean availability
# ---------------------------------------------------------

df_clean["in_stock"] = (
    df_clean["availability_raw"]
    .astype("string")
    .str.contains(
        "In stock",
        case=False,
        na=False
    )
)


# ---------------------------------------------------------
# Handle unexpected numeric values
# ---------------------------------------------------------

# If price or rating cannot be parsed,
# use the median as required by the assignment.

if df_clean["price_gbp"].isna().any():

    price_median = df_clean["price_gbp"].median()

    df_clean["price_gbp"] = (
        df_clean["price_gbp"]
        .fillna(price_median)
    )


if df_clean["rating"].isna().any():

    rating_median = df_clean["rating"].median()

    df_clean["rating"] = (
        df_clean["rating"]
        .fillna(rating_median)
        .round()
        .astype(int)
    )


# ---------------------------------------------------------
# Convert GBP → INR
# ---------------------------------------------------------

df_clean["price_inr"] = (
    df_clean["price_gbp"]
    * FIXED_CONVERSION_RATE
)


# ---------------------------------------------------------
# Keep only required columns
# ---------------------------------------------------------

df_clean = df_clean[
    [
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category_name",
        "availability_raw"
    ]
]


# ---------------------------------------------------------
# Remove rows where essential title/category is missing
# ---------------------------------------------------------

before_drop = len(df_clean)

df_clean = df_clean.dropna(
    subset=[
        "title",
        "category_name"
    ]
)

after_drop = len(df_clean)

if before_drop != after_drop:

    print(
        f"Dropped {before_drop - after_drop} "
        f"rows with missing title/category."
    )


# ---------------------------------------------------------
# Final type conversions
# ---------------------------------------------------------

df_clean["price_gbp"] = df_clean[
    "price_gbp"
].astype(float)

df_clean["price_inr"] = df_clean[
    "price_inr"
].astype(float)

df_clean["rating"] = df_clean[
    "rating"
].astype(int)

df_clean["in_stock"] = df_clean[
    "in_stock"
].astype(bool)


# ---------------------------------------------------------
# PHASE 3: VALIDATION
# ---------------------------------------------------------

print()
print("=" * 60)
print("PHASE 3: VALIDATION")
print("=" * 60)

print(
    f"Final number of books: {len(df_clean)}"
)

print(
    f"Number of categories: "
    f"{df_clean['category_name'].nunique()}"
)

print()
print("Books per category:")

print(
    df_clean["category_name"]
    .value_counts()
)

print()
print("Column data types:")

print(
    df_clean.dtypes
)

print()
print("First 5 cleaned records:")

print(
    df_clean.head()
)


# ---------------------------------------------------------
# Check assignment requirement
# ---------------------------------------------------------

if len(df_clean) < 60:

    raise ValueError(
        "ERROR: Dataset contains fewer than 60 books."
    )

if df_clean["category_name"].nunique() < 3:

    raise ValueError(
        "ERROR: Dataset contains fewer than 3 categories."
    )


# ---------------------------------------------------------
# SAVE CLEANED DATA
# ---------------------------------------------------------

output_file = "books_cleaned.csv"

df_clean.to_csv(
    output_file,
    index=False
)

print()
print("=" * 60)
print("SCRAPING AND CLEANING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    f"Saved cleaned dataset to: {output_file}"
)

print(
    f"Fixed conversion rate: "
    f"1 GBP = {FIXED_CONVERSION_RATE} INR"
)