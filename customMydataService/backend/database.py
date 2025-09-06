import sqlite3
import os

# 데이터베이스 파일 경로 (사용자의 홈 디렉토리에 저장하여 안전하게 관리)
DB_FOLDER = os.path.join(os.path.expanduser('~'), '.customMydataService')
os.makedirs(DB_FOLDER, exist_ok=True)
DB_PATH = os.path.join(DB_FOLDER, 'transactions.db')

def get_db_connection():
    """데이터베이스 연결 객체를 반환합니다."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # 컬럼명으로 접근 가능하게 설정
    return conn

def init_db():
    """데이터베이스와 테이블들을 생성합니다."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. 계좌 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    ''')
    
    # 2. 카테고리 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT NOT NULL UNIQUE,
            major TEXT NOT NULL,
            minor TEXT NOT NULL,
            major_order INTEGER NOT NULL,
            minor_order INTEGER NOT NULL,
            UNIQUE(major, minor)
        )
    ''')

    # 3. 거래 내역 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trans_date TEXT NOT NULL,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            merchant TEXT NOT NULL,
            memo TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            account_id INTEGER,
            category_uuid TEXT,
            FOREIGN KEY (account_id) REFERENCES accounts (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database with new Excel-based schema initialized at: {DB_PATH}")

# --- 아래 함수들은 엑셀 마이그레이션 및 API 구현에 사용될 예정입니다. ---

def find_or_create_account(name):
    """계좌 이름으로 ID를 찾거나, 없으면 새로 생성하고 ID를 반환합니다."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM accounts WHERE name = ?", (name,))
    row = cursor.fetchone()
    if row:
        return row['id']
    else:
        cursor.execute("INSERT INTO accounts (name) VALUES (?)", (name,))
        conn.commit()
        return cursor.lastrowid

def find_or_create_category(major, minor):
    """대분류/소분류로 ID를 찾거나, 없으면 새로 생성하고 ID를 반환합니다."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM categories WHERE major = ? AND minor = ?", (major, minor))
    row = cursor.fetchone()
    if row:
        return row['id']
    else:
        cursor.execute("INSERT INTO categories (major, minor) VALUES (?, ?)", (major, minor))
        conn.commit()
        return cursor.lastrowid

def add_transaction_from_excel(data):
    """엑셀에서 읽어온 한 줄의 데이터를 DB에 추가합니다."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        INSERT INTO transactions (trans_date, type, amount, merchant, memo, account_id, category_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            data['trans_date'],
            data['type'],
            data['amount'],
            data['merchant'],
            data['memo'],
            data['account_id'],
            data['category_id']
        )
    )
    conn.commit()
    conn.close()

def reset_transactions_table():
    """'transactions' 테이블의 모든 데이터를 삭제합니다."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 테이블의 모든 행을 삭제
    cursor.execute('DELETE FROM transactions')
    
    # SQLite의 auto-increment 카운터를 리셋 (선택적이지만 권장)
    # 'transactions' 테이블이 AUTOINCREMENT를 사용하지 않으면 이 라인은 오류를 발생시킬 수 있으므로 try-except로 감쌉니다.
    try:
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='transactions'")
    except sqlite3.OperationalError:
        # sqlite_sequence 테이블이 없거나 해당 항목이 없어도 무시하고 계속 진행
        pass
        
    conn.commit()
    conn.close()
    print("Transactions table has been reset.")

def add_single_transaction(transaction_dict):
    """딕셔너리 형태의 단일 거래내역을 DB에 추가합니다."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 이름으로 ID를 찾거나 생성합니다.
    account_id = find_or_create_account(transaction_dict.get('account')) if transaction_dict.get('account') else None
    category_id = find_or_create_category(transaction_dict.get('majorCategory'), transaction_dict.get('minorCategory')) if transaction_dict.get('majorCategory') and transaction_dict.get('minorCategory') else None

    # ID가 제공되지 않았으면 자동 증가 값 사용
    trans_id = transaction_dict.get('id')

    # 프론트엔드 키를 DB 컬럼에 매핑하여 INSERT
    cursor.execute('''
        INSERT INTO transactions (id, trans_date, type, amount, merchant, memo, account_id, category_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        trans_id,
        transaction_dict['date'],
        transaction_dict['type'],
        transaction_dict['amount'],
        transaction_dict['payee'],
        transaction_dict['memo'],
        account_id,
        category_id
    ))
    
    conn.commit()
    conn.close()

def get_all_transactions_joined():
    """모든 테이블을 JOIN하여 완전한 형태의 거래 내역 목록을 반환합니다."""
    conn = get_db_connection()
    rows = conn.execute('''
        SELECT
            t.id,
            t.trans_date,
            a.name as account,
            t.type,
            c.major as major_category,
            c.minor as minor_category,
            t.amount,
            t.merchant,
            t.memo
        FROM transactions t
        LEFT JOIN accounts a ON t.account_id = a.id
        LEFT JOIN categories c ON t.category_id = c.id
        ORDER BY t.trans_date DESC, t.id ASC
    ''').fetchall()
    conn.close()
    return [dict(row) for row in rows]

def save_all_transactions(transactions_data):
    """프론트엔드에서 받은 전체 거래내역 리스트를 DB에 덮어씁니다."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. 기존 거래내역을 모두 삭제합니다.
    cursor.execute('DELETE FROM transactions')
    # ✨ auto-increment 카운터도 리셋하여 id가 1부터 다시 시작하도록 함
    try:
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='transactions'")
    except sqlite3.OperationalError:
        pass

    # 2. 새로운 거래내역을 하나씩 삽입합니다.
    for t in transactions_data:
        account_id = find_or_create_account(t.get('account')) if t.get('account') else None
        category_id = find_or_create_category(t.get('majorCategory'), t.get('minorCategory')) if t.get('majorCategory') and t.get('minorCategory') else None
        
        # ✨ 프론트의 id를 무시하고 DB가 자동으로 id를 생성하도록 수정
        cursor.execute('''
            INSERT INTO transactions (trans_date, type, amount, merchant, memo, account_id, category_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            t['date'],
            t['type'],
            t['amount'],
            t['payee'],
            t['memo'],
            account_id,
            category_id
        ))
    
    conn.commit()
    conn.close()

def update_minor_category_name(major, old_minor, new_minor):
    """major와 old_minor로 카테고리를 찾아 new_minor로 이름을 변경합니다."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # 먼저 해당 카테고리의 ID를 찾습니다.
    row = cursor.execute('SELECT id FROM categories WHERE major = ? AND minor = ?', (major, old_minor)).fetchone()
    if not row:
        conn.close()
        raise Exception("변경할 카테고리를 찾을 수 없습니다.")
    
    category_id = row['id']
    # 이름을 업데이트합니다.
    cursor.execute('UPDATE categories SET minor = ? WHERE id = ?', (new_minor, category_id))
    conn.commit()
    conn.close()

# ✨ 2. 소분류 삭제 함수 추가
def delete_minor_category_if_unused(major, minor):
    """major와 minor로 카테고리를 찾아, 사용 중이지 않으면 삭제합니다."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # 카테고리 ID 찾기
    row = cursor.execute('SELECT id FROM categories WHERE major = ? AND minor = ?', (major, minor)).fetchone()
    if not row:
        conn.close()
        # 이미 삭제되었을 수 있으므로 오류 대신 성공으로 간주할 수도 있습니다.
        print(f"Warning: Tried to delete non-existent category '{major}/{minor}'")
        return

    category_id = row['id']
    
    # 해당 카테고리를 사용하는 거래내역이 있는지 확인
    usage_exists = cursor.execute('SELECT 1 FROM transactions WHERE category_id = ? LIMIT 1', (category_id,)).fetchone()
    if usage_exists:
        conn.close()
        raise Exception(f"카테고리 '{major}-{minor}'는 거래내역에서 사용 중이므로 삭제할 수 없습니다.")
    
    # 사용 중이지 않으면 삭제
    cursor.execute('DELETE FROM categories WHERE id = ?', (category_id,))
    conn.commit()
    conn.close()

def is_category_in_use(uuid):
    """주어진 uuid가 transactions 테이블에서 사용 중인지 확인합니다."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    usage_exists = cursor.execute(
        'SELECT 1 FROM transactions WHERE category_uuid = ? LIMIT 1', 
        (uuid,)
    ).fetchone()
    
    conn.close()
    return usage_exists is not None