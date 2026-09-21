import sqlite3
import pandas as pd


# ---------------------------------------------------------
# CONNECT TO DATABASE
# ---------------------------------------------------------

DATABASE_FILE = "books_database.db"

conn = sqlite3.connect(DATABASE_FILE)

print("=" * 60)
print("SQL QUERY RESULTS")
print("=" * 60)


# ---------------------------------------------------------
# QUERY 1
# SELECT + WHERE
# ---------------------------------------------------------

query1 = """
SELECT
    title,
    price_gbp,
    rating
FROM books
WHERE rating >= 4
"""

print("\nQUERY 1 - SELECT + WHERE")
print(query1)

result1 = pd.read_sql_query(query1, conn)

print(result1.to_string(index=False))


# ---------------------------------------------------------
# QUERY 2
# ORDER BY + LIMIT
# ---------------------------------------------------------

query2 = """
SELECT
    title,
    price_gbp,
    rating
FROM books
ORDER BY price_gbp DESC
LIMIT 10
"""

print("\nQUERY 2 - ORDER BY + LIMIT")
print(query2)

result2 = pd.read_sql_query(query2, conn)

print(result2.to_string(index=False))


# ---------------------------------------------------------
# QUERY 3
# DISTINCT
# ---------------------------------------------------------

query3 = """
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name
"""

print("\nQUERY 3 - DISTINCT")
print(query3)

result3 = pd.read_sql_query(query3, conn)

print(result3.to_string(index=False))


# ---------------------------------------------------------
# QUERY 4
# IN
# ---------------------------------------------------------

query4 = """
SELECT
    title,
    price_gbp,
    rating
FROM books
WHERE rating IN (4, 5)
"""

print("\nQUERY 4 - IN")
print(query4)

result4 = pd.read_sql_query(query4, conn)

print(result4.to_string(index=False))


# ---------------------------------------------------------
# QUERY 5
# BETWEEN
# ---------------------------------------------------------

query5 = """
SELECT
    title,
    price_gbp,
    price_inr
FROM books
WHERE price_gbp BETWEEN 20 AND 40
"""

print("\nQUERY 5 - BETWEEN")
print(query5)

result5 = pd.read_sql_query(query5, conn)

print(result5.to_string(index=False))


# ---------------------------------------------------------
# QUERY 6
# JOIN
# ---------------------------------------------------------

query6 = """
SELECT
    c.category_name,
    b.title,
    b.rating,
    b.price_gbp
FROM books b
JOIN categories c
    ON b.category_id = c.category_id
ORDER BY
    c.category_name,
    b.rating DESC,
    b.title
"""

print("\nQUERY 6 - JOIN")
print(query6)

result6 = pd.read_sql_query(query6, conn)

print(result6.to_string(index=False))


# ---------------------------------------------------------
# SAVE QUERY OUTPUTS
# ---------------------------------------------------------

with open("query_outputs.txt", "w", encoding="utf-8") as file:

    file.write("=" * 60 + "\n")
    file.write("MODULE 1 SQL QUERY OUTPUTS\n")
    file.write("=" * 60 + "\n\n")

    file.write("QUERY 1 - SELECT + WHERE\n")
    file.write(query1)
    file.write("\nOUTPUT:\n")
    file.write(result1.to_string(index=False))
    file.write("\n\n")

    file.write("QUERY 2 - ORDER BY + LIMIT\n")
    file.write(query2)
    file.write("\nOUTPUT:\n")
    file.write(result2.to_string(index=False))
    file.write("\n\n")

    file.write("QUERY 3 - DISTINCT\n")
    file.write(query3)
    file.write("\nOUTPUT:\n")
    file.write(result3.to_string(index=False))
    file.write("\n\n")

    file.write("QUERY 4 - IN\n")
    file.write(query4)
    file.write("\nOUTPUT:\n")
    file.write(result4.to_string(index=False))
    file.write("\n\n")

    file.write("QUERY 5 - BETWEEN\n")
    file.write(query5)
    file.write("\nOUTPUT:\n")
    file.write(result5.to_string(index=False))
    file.write("\n\n")

    file.write("QUERY 6 - JOIN\n")
    file.write(query6)
    file.write("\nOUTPUT:\n")
    file.write(result6.to_string(index=False))
    file.write("\n\n")


# ---------------------------------------------------------
# CLOSE DATABASE
# ---------------------------------------------------------

conn.close()

print("\n" + "=" * 60)
print("SQL QUERIES COMPLETED")
print("=" * 60)

print("Query outputs saved to: query_outputs.txt")