import sqlite3

# def execute_sql_file(db_conn, sql_file_path):
#     try:
#         with open(sql_file_path, 'r') as sql_file:
#             sql_script = sql_file.read()
#             db_conn.executescript(sql_script)
#         print("SQL script executed successfully.")
#     except Exception as e:
#         print(f"Error executing SQL script: {e}")

def main():
    db_path = "uniref50.db"  # Path to your SQLite database file
    sql_file_path = "/Users/ambroseling/Desktop/NucleAIse/nucleaise/preprocessing/uniref50.sql"  # Path to your SQL file

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Execute the SQL script contained in the file
    execute_sql_file(conn, sql_file_path)

    # Don't forget to commit the changes and close the connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
