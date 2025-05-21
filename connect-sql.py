import sqlite3
import pandas as pd

conn = sqlite3.connect("./data/penguins.db")

# SELECT 쿼리 결과를 DataFrame으로 읽기
df = pd.read_sql_query("SELECT * FROM penguins LIMIT 5;", conn)
print(df.head())

df.info()

df2 = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "age": [25, 30]
})

conn.close()