import sqlite3
import pandas as pd


# ---------------------------------------------------------
# CONNECT TO DATABASE
# ---------------------------------------------------------

conn = sqlite3.connect("books_database.db")


# ---------------------------------------------------------
# READ TWO SQL RESULTS USING pd.read_sql()
# ---------------------------------------------------------

query1 = """
SELECT
    title,
    price_gbp,
    rating
FROM books
WHERE rating >= 4
ORDER BY rating DESC
"""

query2 = """
SELECT
    title,
    price_gbp,
    price_inr
FROM books
ORDER BY price_gbp DESC
LIMIT 10
"""

df_query1 = pd.read_sql(query1, conn)

df_query2 = pd.read_sql(query2, conn)


print("=" * 60)
print("PANDAS: SQL QUERY RESULTS")
print("=" * 60)

print("\nResult 1:")
print(df_query1.to_string(index=False))

print("\nResult 2:")
print(df_query2.to_string(index=False))


# ---------------------------------------------------------
# READ TABLES INTO DATAFRAMES
# ---------------------------------------------------------

books_df = pd.read_sql(
    "SELECT * FROM books",
    conn
)

categories_df = pd.read_sql(
    "SELECT * FROM categories",
    conn
)


# ---------------------------------------------------------
# REPRODUCE JOIN USING pd.merge()
# ---------------------------------------------------------

merged_df = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)


# Select the same columns as our SQL JOIN

merge_result = merged_df[
    [
        "category_name",
        "title",
        "rating",
        "price_gbp"
    ]
].copy()


# Same ordering as SQL JOIN

merge_result = merge_result.sort_values(
    by=[
        "category_name",
        "rating",
        "title"
    ],
    ascending=[
        True,
        False,
        True
    ]
).reset_index(drop=True)


# ---------------------------------------------------------
# SQL JOIN RESULT
# ---------------------------------------------------------

sql_join_query = """
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

sql_join_result = pd.read_sql(
    sql_join_query,
    conn
).reset_index(drop=True)


# ---------------------------------------------------------
# COMPARE SQL JOIN AND pd.merge()
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SQL JOIN RESULT")
print("=" * 60)

print(
    sql_join_result.head(10).to_string(index=False)
)


print("\n" + "=" * 60)
print("pd.merge() RESULT")
print("=" * 60)

print(
    merge_result.head(10).to_string(index=False)
)


# ---------------------------------------------------------
# CHECK EQUIVALENCE
# ---------------------------------------------------------

same_result = sql_join_result.equals(
    merge_result
)

print("\n" + "=" * 60)
print("EQUIVALENCE CHECK")
print("=" * 60)

print(
    "SQL JOIN and pd.merge() produce "
    f"equivalent results: {same_result}"
)


# ---------------------------------------------------------
# SAVE VERIFICATION OUTPUT
# ---------------------------------------------------------

with open(
    "pandas_verification.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PANDAS VERIFICATION\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        "Query 1 using pd.read_sql():\n"
    )

    file.write(
        df_query1.to_string(index=False)
    )

    file.write("\n\n")

    file.write(
        "Query 2 using pd.read_sql():\n"
    )

    file.write(
        df_query2.to_string(index=False)
    )

    file.write("\n\n")

    file.write(
        "SQL JOIN result:\n"
    )

    file.write(
        sql_join_result.to_string(index=False)
    )

    file.write("\n\n")

    file.write(
        "pd.merge() result:\n"
    )

    file.write(
        merge_result.to_string(index=False)
    )

    file.write("\n\n")

    file.write(
        "SQL JOIN and pd.merge() equivalent: "
        f"{same_result}\n"
    )


# ---------------------------------------------------------
# CLOSE DATABASE
# ---------------------------------------------------------

conn.close()

print(
    "\nVerification saved to: "
    "pandas_verification.txt"
)