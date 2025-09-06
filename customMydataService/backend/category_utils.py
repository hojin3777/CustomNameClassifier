import database
import os
import json

DEFAULT_CATEGORIES = {
    "계좌": ["계좌1", "계좌2", "계좌3"],
    "고정수입": ["정기급여", "금융수입", "용돈"],
    "유동수입": ["상여금", "사업수입", "금융수입", "용돈", "기타수입"],
    "이체분류": ["내계좌이체", "이체", "저축", "현금", "투자"],
    "식비": ["외식", "식재료", "배달", "포장"],
    "카페/간식": ["커피/음료", "베이커리", "디저트/빵", "아이스크림/빙수"],
    "외출/주점": ["노래방", "PC방", "당구장", "만화방", "주점"],
    "생활": ["생필품", "편의점", "마트", "세탁", "지역화폐충전", "가구/가전", "문구류", "전자제품"],
    "온라인쇼핑": ["서비스구독", "앱스토어", "인터넷쇼핑", "수수료"],
    "패션/쇼핑": ["옷", "신발", "액세서리", "백화점"],
    "뷰티/미용": ["화장품", "헤어샵", "미용관리", "미용용품"],
    "교통": ["택시", "대중교통", "시외버스", "철도", "전동킥보드", "렌터카", "항공"],
    "자동차": ["주유", "주차", "세차", "통행료", "정비/수리", "자동차보험", "대리운전", "과태료"],
    "주거/통신": ["휴대폰", "인터넷", "월세", "관리비", "가스비", "전기세"],
    "의료/건강": ["약국", "병원", "건강/보조식품", "운동"],
    "금융": ["보험", "증권/투자", "카드", "이자/대출", "세금/과태료"],
    "문화/여가": ["영화", "도서", "게임", "공연", "전시/관람/체험", "취미", "테마파크", "기타"],
    "여행/숙박": ["숙박비", "관광", "교통비", "기념품", "여행용품"],
    "교육/학습": ["수업료", "시험료", "책"],
    "경조/선물": ["축의금", "부조금", "선물", "회비"]
}

# ✨ 1. DB가 비어있을 때 기본값으로 채우는 함수
def initialize_default_categories():
    """accounts와 categories 테이블이 비어있으면 기본값으로 초기화합니다."""
    conn = database.get_db_connection()
    # 테이블이 비어있는지 확인
    accounts_count = conn.execute('SELECT COUNT(*) FROM accounts').fetchone()[0]
    categories_count = conn.execute('SELECT COUNT(*) FROM categories').fetchone()[0]
    
    # 두 테이블이 모두 비어있을 때만 실행
    if accounts_count == 0 and categories_count == 0:
        print("Initializing default accounts and categories in the database...")
        major_order_index = 0
        major_code_char_code = ord('A')
        for major, minors in DEFAULT_CATEGORIES.items():
            if major == "계좌":
                for name in minors:
                    conn.execute('INSERT INTO accounts (name) VALUES (?)', (name,))
            else:
                major_code = chr(major_code_char_code)
                minor_order_index = 0
                minor_num = 1
                for minor in minors:
                    uuid = f"{major_code}{minor_num}"
                    conn.execute(
                        'INSERT INTO categories (uuid, major, minor, major_order, minor_order) VALUES (?, ?, ?, ?, ?)',
                        (uuid, major, minor, major_order_index, minor_order_index)
                    )
                    minor_order_index += 1
                    minor_num += 1
                major_order_index += 1
                major_code_char_code += 1
        conn.commit()
        print("Default data initialization complete.")
    
    conn.close()

# ✨ 2. DB에서 데이터를 불러와 프론트엔드 형식으로 변환하는 함수
def load_categories_from_db():
    """DB에서 계좌와 카테고리를 읽어 프론트엔드 형식으로 그룹화하여 반환합니다."""
    conn = database.get_db_connection()
    # 계좌 데이터 불러오기 (ID 순)
    accounts_cursor = conn.execute('SELECT name FROM accounts ORDER BY id')
    account_minors = [row['name'] for row in accounts_cursor.fetchall()]
    
    # ✨ 카테고리 데이터 불러오기 (대분류의 첫 등장 ID, 그 다음 소분류 ID 순으로 정렬)
    categories_cursor = conn.execute('''
        SELECT uuid, major, minor 
        FROM categories 
        ORDER BY major_order, minor_order
    ''').fetchall()
    conn.close()

    final_data = [{"major": "계좌", "minors": account_minors}]
    if categories_cursor:
        # 마지막으로 처리한 대분류를 추적
        last_major = None
        for row in categories_cursor:
            major, minor, uuid = row['major'], row['minor'], row['uuid']
            minor_obj = {"name": minor, "uuid": uuid}
            # 새로운 대분류를 만나면, final_data에 새 객체를 추가
            if major != last_major:
                final_data.append({"major": major, "minors": [minor_obj]})
                last_major = major
            # 같은 대분류이면, 마지막 객체의 minors 배열에 소분류만 추가
            else:
                final_data[-1]["minors"].append(minor)
            
    return final_data

# ✨ 3. 프론트엔드에서 받은 데이터로 DB를 업데이트하는 함수
def save_categories_to_db(data):
    conn = database.get_db_connection()
    cursor = conn.cursor()

    # --- 1. 계좌 업데이트 ---
    # 프론트에서 받은 계좌 목록
    frontend_accounts_data = next((item for item in data if item['major'] == '계좌'), None)
    frontend_accounts = set(frontend_accounts_data['minors']) if frontend_accounts_data else set()

    # DB의 현재 계좌 목록
    db_accounts_cursor = cursor.execute('SELECT name FROM accounts').fetchall()
    db_accounts = {row['name'] for row in db_accounts_cursor}

    # 삭제/추가할 계좌 처리
    accounts_to_delete = db_accounts - frontend_accounts
    if accounts_to_delete:
        delete_tuple = tuple(accounts_to_delete)
        cursor.execute(f"DELETE FROM accounts WHERE name IN ({','.join('?'*len(delete_tuple))})", delete_tuple)
    
    accounts_to_add = frontend_accounts - db_accounts
    for name in accounts_to_add:
        cursor.execute('INSERT INTO accounts (name) VALUES (?)', (name,))

    # --- 2. 카테고리 지능형 업데이트 ---
    db_categories_raw = cursor.execute('SELECT id, uuid, major, minor FROM categories').fetchall()
    db_categories_by_uuid = {row['uuid']: row for row in db_categories_raw}
    
    major_codes = {row['uuid'][0] for row in db_categories_raw}
    next_major_code_char_code = ord('A')
    while chr(next_major_code_char_code) in major_codes:
        next_major_code_char_code += 1

    major_to_code_map = {}
    for uuid_val, row in db_categories_by_uuid.items():
        if row['major'] not in major_to_code_map:
            major_to_code_map[row['major']] = uuid_val[0]

    frontend_uuids = set()
    major_order_index = 0
    # '계좌'를 제외한 카테고리만 순회
    for item in [d for d in data if d.get('major') != '계좌']:
        major = item['major']
        if not major: continue

        if major not in major_to_code_map:
            major_code = chr(next_major_code_char_code)
            major_to_code_map[major] = major_code
            next_major_code_char_code += 1
        else:
            major_code = major_to_code_map[major]
        
        next_minor_num = 1
        for uuid_key in db_categories_by_uuid:
            if uuid_key.startswith(major_code):
                try:
                    num = int(uuid_key[1:])
                    if num >= next_minor_num:
                        next_minor_num = num + 1
                except (ValueError, IndexError):
                    continue

        minor_order_index = 0
        for minor_item in item.get('minors', []):
            # ✨ 여기서 minor_item이 객체임을 보장합니다.
            minor_name = minor_item.get('name')
            minor_uuid = minor_item.get('uuid')

            if minor_uuid and minor_uuid in db_categories_by_uuid:
                frontend_uuids.add(minor_uuid)
                cursor.execute(
                    'UPDATE categories SET major = ?, minor = ?, major_order = ?, minor_order = ? WHERE uuid = ?',
                    (major, minor_name, major_order_index, minor_order_index, minor_uuid)
                )
            else:
                new_uuid = f"{major_code}{next_minor_num}"
                frontend_uuids.add(new_uuid)
                cursor.execute(
                    'INSERT INTO categories (uuid, major, minor, major_order, minor_order) VALUES (?, ?, ?, ?, ?)',
                    (new_uuid, major, minor_name, major_order_index, minor_order_index)
                )
                next_minor_num += 1
            minor_order_index += 1
        major_order_index += 1

    uuids_to_delete = set(db_categories_by_uuid.keys()) - frontend_uuids
    if uuids_to_delete:
        delete_tuple = tuple(uuids_to_delete)
        cursor.execute(f"DELETE FROM categories WHERE uuid IN ({','.join('?'*len(delete_tuple))})", delete_tuple)

    conn.commit()
    conn.close()