import sqlite3
import os

# 데이터베이스 파일 경로 (사용자의 홈 디렉토리에 저장하여 안전하게 관리)
DB_FOLDER = os.path.join(os.path.expanduser('~'), '.customMydataService')
os.makedirs(DB_FOLDER, exist_ok=True)
DB_PATH = os.path.join(DB_FOLDER, 'mydata.db')

def get_db_connection():
    """데이터베이스 연결 객체를 반환합니다."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")  # 외래 키 제약 조건 활성화
    conn.row_factory = sqlite3.Row # 컬럼명으로 접근 가능하게 설정
    return conn

def init_db():
    """데이터베이스와 테이블들을 생성합니다."""
    print(f"Initializing database at: {DB_PATH}")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # --- 1. 계좌 테이블 (accounts) ---
    # 역할: 계좌의 이름과 표시 순서를 관리합니다.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            display_order INTEGER NOT NULL
        )
    ''')
    # --- 2. 대분류 테이블 (major_categories) ---
    # 역할: 카테고리 그룹의 이름과 표시 순서를 관리합니다.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS major_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            display_order INTEGER NOT NULL
        )
    ''')
    # --- 3. 소분류 테이블 (minor_categories) ---
    # 역할: 실제 카테고리 항목의 이름, 순서, 그리고 어떤 대분류에 속해있는지를 관리합니다.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS minor_categories (
            uuid TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            major_category_id INTEGER NOT NULL,
            display_order INTEGER NOT NULL,
            FOREIGN KEY (major_category_id) REFERENCES major_categories (id) ON DELETE CASCADE
        )
    ''')
    # ON DELETE CASCADE: 대분류가 삭제되면, 거기에 속한 모든 소분류도 자동으로 함께 삭제됩니다.

    # --- 4. 거래 내역 테이블 (transactions) ---
    # 역할: 모든 거래 기록을 저장합니다. 계좌와 소분류를 외래 키로 참조합니다.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_date TEXT NOT NULL,
            account_id INTEGER NOT NULL,                   
            type TEXT NOT NULL,
            minor_category_uuid TEXT NOT NULL,
            amount INTEGER NOT NULL,
            merchant TEXT NOT NULL,
            memo TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_bold INTEGER DEFAULT 0,
            flag_color_id INTEGER DEFAULT 0,
            highlight_color_id INTEGER DEFAULT 0,
            background_color_id INTEGER DEFAULT 0,
            FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE SET NULL,
            FOREIGN KEY (minor_category_uuid) REFERENCES minor_categories (uuid) ON DELETE SET NULL
        )
    ''')

    # --- 5. 카테고리 매핑 테이블 (category_mappings) ---
    # 역할: BERT 모델의 출력을 사용자의 소분류에 매핑합니다.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS category_mappings (
            bert_output_id INTEGER PRIMARY KEY,
            bert_output_name TEXT NOT NULL UNIQUE,
            minor_category_uuid TEXT,
            FOREIGN KEY (minor_category_uuid) REFERENCES minor_categories (uuid) ON DELETE SET NULL
        )
    ''')
    # ON DELETE SET NULL: 매핑된 소분류가 삭제되면, 이 테이블의 해당 항목은 NULL로 자동 변경됩니다. (매핑 해제 효과)

    conn.commit()
    conn.close()
    print(f"Database and tables created successfully created at: {DB_PATH}")

def is_account_in_use(account_id):
    """특정 계좌가 거래내역에서 사용 중인지 확인합니다."""
    conn = get_db_connection()
    count = conn.execute('SELECT COUNT(*) FROM transactions WHERE account_id = ?', (account_id,)).fetchone()[0]
    conn.close()
    return count > 0

def is_minor_category_in_use(minor_uuid):
    """특정 소분류가 거래내역에서 사용 중인지 확인합니다."""
    conn = get_db_connection()
    count = conn.execute('SELECT COUNT(*) FROM transactions WHERE minor_category_uuid = ?', (minor_uuid,)).fetchone()[0]
    conn.close()
    return count > 0

def reset_all_transactions():
    """거래내역 테이블을 초기화합니다."""
    conn = get_db_connection()
    conn.execute('DELETE FROM transactions')
    conn.execute('DELETE FROM sqlite_sequence WHERE name="transactions"')  # AUTOINCREMENT 초기화
    conn.commit()
    conn.close()
    print("All transactions have been reset.")