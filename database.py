def initialize_database():
    conn = sqlite3.connect('temp3.db')
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(chat_bot)")
    columns = [info[1] for info in cursor.fetchall()]
    if "user_input" in columns and "chatbot_id" not in columns:
       
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS new_chat_bot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chatbot_id TEXT,
                user_input TEXT,
                chatbot_response TEXT
            )
        ''')
        conn.commit()

        cursor.execute('''
            INSERT INTO new_chat_bot (user_input, chatbot_response)
            SELECT user_input, chatbot_response FROM chat_bot
        ''')
        conn.commit()

        cursor.execute("DROP TABLE chat_bot")
        cursor.execute("ALTER TABLE new_chat_bot RENAME TO chat_bot")
        conn.commit()

    else:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_bot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chatbot_id TEXT,
                user_input TEXT,
                chatbot_response TEXT
            )
        ''')
        conn.commit()

    return conn, cursor


def clear_history(conn, cursor, chatbot_id):
    cursor.execute('DELETE FROM chat_bot WHERE chatbot_id = ?', (chatbot_id,))
    conn.commit()
