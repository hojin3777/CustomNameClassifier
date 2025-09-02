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
            major TEXT NOT NULL,
            minor TEXT NOT NULL,
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
            category_id INTEGER,
            FOREIGN KEY (account_id) REFERENCES accounts (id),
            FOREIGN KEY (category_id) REFERENCES categories (id)
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
        ORDER BY t.trans_date DESC, t.id DESC
    ''').fetchall()
    conn.close()
    return [dict(row) for row in rows]