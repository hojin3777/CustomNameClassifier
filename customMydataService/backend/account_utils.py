import database

def load_accounts():
    """DB에서 모든 계좌 목록을 불러와 리스트로 반환합니다."""
    conn = database.get_db_connection()
    accounts_cursor = conn.execute('SELECT name FROM accounts ORDER BY id').fetchall()
    conn.close()
    return [row['name'] for row in accounts_cursor]

def save_accounts(account_list):
    """프론트엔드에서 받은 계좌 리스트를 DB에 덮어씁니다. (전체 삭제 후 재삽입)"""
    conn = database.get_db_connection()
    cursor = conn.cursor()

    # 1. 기존 계좌 모두 삭제
    cursor.execute('DELETE FROM accounts')
    # 2. auto-increment 리셋 (id를 1부터 다시 시작)
    try:
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='accounts'")
    except database.sqlite3.OperationalError:
        pass

    # 3. 새로운 리스트 순서대로 삽입
    for name in account_list:
        cursor.execute('INSERT INTO accounts (name) VALUES (?)', (name,))
    
    conn.commit()
    conn.close()