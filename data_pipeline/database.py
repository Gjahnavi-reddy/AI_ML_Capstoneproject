import sqlite3
import pandas as pd


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

CSV_FILE = "books_cleaned.csv"
DATABASE_FILE = "books_database.db"


# ---------------------------------------------------------
# LOAD CLEANED DATA
# ---------------------------------------------------------

print("=" * 60)
print("LOADING CLEANED DATA")
print("=" * 60)

df = pd.read_csv(CSV_FILE)

print(f"Records loaded: {len(df)}")
print(f"Categories: {df['category_name'].nunique()}")


# ---------------------------------------------------------
# CONNECT TO SQLITE
# ---------------------------------------------------------

conn = sqlite3.connect(DATABASE_FILE)

cursor = conn.cursor()

# Enable foreign key constraints
cursor.execute("PRAGMA foreign_keys = ON;")


# ---------------------------------------------------------
# CREATE TABLES
# ---------------------------------------------------------

cursor.execute("DROP TABLE IF EXISTS books")
cursor.execute("DROP TABLE IF EXISTS categories")


cursor.execute("""
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
)
""")


cursor.execute("""
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")


# ---------------------------------------------------------
# INSERT CATEGORIES
# ---------------------------------------------------------

categories = df["category_name"].unique()

for category in categories:

    cursor.execute(
        """
        INSERT INTO categories (category_name)
        VALUES (?)
        """,
        (category,)
    )


# ---------------------------------------------------------
# CREATE CATEGORY → ID MAPPING
# ---------------------------------------------------------

category_map = {}

cursor.execute(
    "SELECT category_id, category_name FROM categories"
)

for category_id, category_name in cursor.fetchall():

    category_map[category_name] = category_id


# ---------------------------------------------------------
# INSERT BOOKS
# ---------------------------------------------------------

for _, row in df.iterrows():

    category_id = category_map[
        row["category_name"]
    ]

    cursor.execute(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            int(row["in_stock"]),
            category_id
        )
    )


# ---------------------------------------------------------
# SAVE DATABASE
# ---------------------------------------------------------

conn.commit()


# ---------------------------------------------------------
# VERIFY DATABASE
# ---------------------------------------------------------

print()
print("=" * 60)
print("DATABASE VERIFICATION")
print("=" * 60)


cursor.execute("SELECT COUNT(*) FROM books")

book_count = cursor.fetchone()[0]

print(f"Books in database: {book_count}")


cursor.execute("SELECT COUNT(*) FROM categories")

category_count = cursor.fetchone()[0]

print(f"Categories in database: {category_count}")


print()
print("Categories:")

cursor.execute(
    """
    SELECT category_id, category_name
    FROM categories
    ORDER BY category_id
    """
)

for row in cursor.fetchall():

    print(row)


print()
print("Sample books:")

cursor.execute(
    """
    SELECT
        book_id,
        title,
        price_gbp,
        price_inr,
        rating,
        in_stock,
        category_id
    FROM books
    LIMIT 5
    """
)

for row in cursor.fetchall():

    print(row)


# ---------------------------------------------------------
# CLOSE DATABASE
# ---------------------------------------------------------

conn.close()

print()
print("=" * 60)
print("DATABASE CREATED SUCCESSFULLY")
print("=" * 60)

print(f"Database file: {DATABASE_FILE}")