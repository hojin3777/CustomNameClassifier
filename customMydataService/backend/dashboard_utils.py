import database
from datetime import datetime

def get_dashboard_trend_range():
    """데이터베이스에서 저장된 대시보드 기간 설정을 불러옵니다."""
    range_str = database.get_setting('dashboard_trend_range')
    if range_str:
        try:
            # "2024-01,2025-01" 같은 문자열을 ['2024-01', '2025-01'] 리스트로 변환
            parts = range_str.split(',')
            if len(parts) == 2:
                return parts
        except Exception:
            return None
    return None

def set_dashboard_trend_range(range_list):
    """대시보드 기간 설정을 데이터베이스에 저장합니다."""
    if isinstance(range_list, list) and len(range_list) == 2:
        # ['2024-01', '2025-01'] 리스트를 "2024-01,2025-01" 문자열로 변환
        range_str = f"{range_list[0]},{range_list[1]}"
        database.set_setting('dashboard_trend_range', range_str)

def get_monthly_summary(start_month_str=None, end_month_str=None):
    """
    지정된 기간 동안의 월별 상세 수입/지출을 계산합니다.
    기간이 지정되지 않으면 전체 기간을 반환합니다.
    """
    conn = database.get_db_connection()
    
    query = """
        SELECT
            strftime('%Y-%m', t.transaction_date) AS month,
            SUM(CASE WHEN mc.name = '고정수입' THEN t.amount ELSE 0 END) AS fixed_income,
            SUM(CASE WHEN mc.name = '유동수입' THEN t.amount ELSE 0 END) AS variable_income,
            SUM(CASE WHEN t.type = '고정지출' THEN ABS(t.amount) ELSE 0 END) AS fixed_expense,
            SUM(CASE WHEN t.type = '반고정지출' THEN ABS(t.amount) ELSE 0 END) AS semi_fixed_expense,
            SUM(CASE WHEN t.type = '유동지출' THEN ABS(t.amount) ELSE 0 END) AS variable_expense
        FROM transactions t
        LEFT JOIN minor_categories mnc ON t.minor_category_uuid = mnc.uuid
        LEFT JOIN major_categories mc ON mnc.major_category_id = mc.id
    """
    
    params = []
    if start_month_str and end_month_str:
        query += " WHERE strftime('%Y-%m', t.transaction_date) BETWEEN ? AND ?"
        params.extend([start_month_str, end_month_str])

    query += " GROUP BY month ORDER BY month ASC"
    
    cursor = conn.execute(query, params)
    summary = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return summary

def get_available_months():
    """DB에 있는 모든 거래내역의 월 목록(YYYY-MM)을 반환합니다."""
    conn = database.get_db_connection()
    query = "SELECT DISTINCT strftime('%Y-%m', transaction_date) as month FROM transactions ORDER BY month ASC"
    cursor = conn.execute(query)
    months = [row['month'] for row in cursor.fetchall()]
    conn.close()
    return months

def get_monthly_detail_summary(year, month):
    """지정된 월의 상세 수입/지출 내역을 계산합니다."""
    conn = database.get_db_connection()
    month_str = f"{year}-{month:02d}"
    
    query = """--sql
        SELECT
            SUM(CASE WHEN mc.name = '고정수입' THEN t.amount ELSE 0 END) AS fixed_income,
            SUM(CASE WHEN mc.name = '유동수입' THEN t.amount ELSE 0 END) AS variable_income,
            SUM(CASE WHEN t.type = '고정지출' THEN ABS(t.amount) ELSE 0 END) AS fixed_expense,
            SUM(CASE WHEN t.type = '반고정지출' THEN ABS(t.amount) ELSE 0 END) AS semi_fixed_expense,
            SUM(CASE WHEN t.type = '유동지출' THEN ABS(t.amount) ELSE 0 END) AS variable_expense
        FROM transactions t
        LEFT JOIN minor_categories mnc ON t.minor_category_uuid = mnc.uuid
        LEFT JOIN major_categories mc ON mnc.major_category_id = mc.id
        WHERE strftime('%Y-%m', t.transaction_date) = ?
    """
    
    cursor = conn.execute(query, (month_str,))
    # fetchone()은 결과가 없을 때 None을 반환할 수 있으므로, 기본값을 설정합니다.
    row = cursor.fetchone()
    conn.close()

    if row and any(row):
        summary = dict(row)
    else:
        # 해당 월에 데이터가 전혀 없을 경우 0으로 채워진 기본 구조를 반환합니다.
        summary = {
            "fixed_income": 0,
            "variable_income": 0,
            "fixed_expense": 0,
            "semi_fixed_expense": 0,
            "variable_expense": 0
        }
        
    return summary

def get_category_spending(year, month):
    """지정된 월의 대분류별 지출을 계산합니다."""
    conn = database.get_db_connection()
    query = """--sql
        SELECT
            mc.name,
            SUM(ABS(t.amount)) as value
        FROM transactions t
        JOIN minor_categories mnc ON t.minor_category_uuid = mnc.uuid
        JOIN major_categories mc ON mnc.major_category_id = mc.id
        WHERE strftime('%Y', t.transaction_date) = ? 
          AND strftime('%m', t.transaction_date) = ?
          AND mc.name NOT IN ('고정수입', '유동수입', '이체분류')
        GROUP BY mc.id, mc.name
        ORDER BY value DESC
    """
    cursor = conn.execute(query, (str(year), f'{month:02d}'))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    total_spending = sum(row['value'] for row in rows)
    if total_spending == 0:
        return []

    result = [
        {
            'name': row['name'],
            'value': row['value'],
            'percentage': (row['value'] / total_spending) * 100
        }
        for row in rows
    ]
    
    return result

def get_account_balances():
    """모든 계좌의 잔액을 계산합니다."""
    conn = database.get_db_connection()
    query = """
        SELECT 
            a.name AS account_name,
            COALESCE(SUM(t.amount), 0) AS balance
        FROM accounts a
        LEFT JOIN transactions t ON a.id = t.account_id
        WHERE a.name NOT LIKE '(exp)%' AND a.name NOT LIKE '(숨김)%'
        GROUP BY a.id, a.name, a.display_order
        ORDER BY a.display_order ASC
    """
    cursor = conn.execute(query)
    balances = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return balances

def get_category_treemap(year, month):
    """
    지정된 월의 대분류-소분류별 지출 비율을 계층 구조로 반환합니다.
    """
    conn = database.get_db_connection()
    query = """
        SELECT
            mc.id AS major_id,
            mc.name AS major_name,
            mnc.uuid AS minor_uuid,
            mnc.name AS minor_name,
            SUM(ABS(t.amount)) as value
        FROM transactions t
        JOIN minor_categories mnc ON t.minor_category_uuid = mnc.uuid
        JOIN major_categories mc ON mnc.major_category_id = mc.id
        WHERE strftime('%Y', t.transaction_date) = ?
          AND strftime('%m', t.transaction_date) = ?
          AND mc.name NOT IN ('고정수입', '유동수입', '이체분류')
        GROUP BY mc.id, mc.name, mnc.uuid, mnc.name
        ORDER BY mc.id, value DESC
    """
    cursor = conn.execute(query, (str(year), f'{month:02d}'))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    # 계층 구조로 변환
    major_map = {}
    for row in rows:
        major_id = row['major_id']
        if major_id not in major_map:
            major_map[major_id] = {
                'name': row['major_name'],
                'value': 0,
                'children': []
            }
        major_map[major_id]['children'].append({
            'name': row['minor_name'],
            'value': row['value']
        })
        major_map[major_id]['value'] += row['value']

    # 최상위 노드 리스트로 변환
    result = sorted(list(major_map.values()), key=lambda x: x['value'], reverse=True)
    return result

def get_top_spending_categories(start_month_str, end_month_str):
    """
    지정된 기간 동안 지출액 기준 및 지출 빈도 기준 상위 10개 소분류와
    각 소분류에 대한 상위 10개 거래처 상세 내역을 반환합니다.
    """
    conn = database.get_db_connection()
    params = (start_month_str, end_month_str)
    
    # 1. 지출액 기준 TOP 10 소분류 UUID 조회
    query_top_amount_categories = """--sql
        SELECT mnc.uuid, mnc.name, SUM(ABS(t.amount)) as value
        FROM transactions t
        JOIN minor_categories mnc ON t.minor_category_uuid = mnc.uuid
        JOIN major_categories mc ON mnc.major_category_id = mc.id
        WHERE strftime('%Y-%m', t.transaction_date) BETWEEN ? AND ?
          AND mc.name NOT IN ('고정수입', '유동수입', '이체분류')
        GROUP BY mnc.uuid, mnc.name
        ORDER BY value DESC
        LIMIT 10
    """
    top_amount_categories = conn.execute(query_top_amount_categories, params).fetchall()

    # 2. 지출 빈도 기준 TOP 10 소분류 UUID 조회
    query_top_freq_categories = """--sql
        SELECT mnc.uuid, mnc.name, COUNT(t.id) as value
        FROM transactions t
        JOIN minor_categories mnc ON t.minor_category_uuid = mnc.uuid
        JOIN major_categories mc ON mnc.major_category_id = mc.id
        WHERE strftime('%Y-%m', t.transaction_date) BETWEEN ? AND ?
          AND mc.name NOT IN ('고정수입', '유동수입', '이체분류')
        GROUP BY mnc.uuid, mnc.name
        ORDER BY value DESC
        LIMIT 10
    """
    top_freq_categories = conn.execute(query_top_freq_categories, params).fetchall()

    # 3. 각 카테고리별 상세 내역 조회 및 결과 조합
    top_by_amount = []
    for category in top_amount_categories:
        details_query = """--sql
            SELECT merchant as name, SUM(ABS(amount)) as value, COUNT(id) as count
            FROM transactions
            WHERE minor_category_uuid = ? AND strftime('%Y-%m', transaction_date) BETWEEN ? AND ?
            GROUP BY merchant
            ORDER BY value DESC
            LIMIT 10
        """
        details_cursor = conn.execute(details_query, (category['uuid'], start_month_str, end_month_str))
        details = [dict(row) for row in details_cursor.fetchall()]

        total_count_query = "SELECT COUNT(DISTINCT merchant) FROM transactions WHERE minor_category_uuid = ? AND strftime('%Y-%m', transaction_date) BETWEEN ? AND ?"
        total_count = conn.execute(total_count_query, (category['uuid'], start_month_str, end_month_str)).fetchone()[0]
        
        top_by_amount.append({
            "name": category['name'],
            "value": category['value'],
            "details": {"items": details, "total_count": total_count}
        })

    top_by_frequency = []
    for category in top_freq_categories:
        details_query = """--sql
            SELECT merchant as name, COUNT(id) as value
            FROM transactions
            WHERE minor_category_uuid = ? AND strftime('%Y-%m', transaction_date) BETWEEN ? AND ?
            GROUP BY merchant
            ORDER BY value DESC
            LIMIT 10
        """
        details_cursor = conn.execute(details_query, (category['uuid'], start_month_str, end_month_str))
        details = [dict(row) for row in details_cursor.fetchall()]

        total_count_query = "SELECT COUNT(DISTINCT merchant) FROM transactions WHERE minor_category_uuid = ? AND strftime('%Y-%m', transaction_date) BETWEEN ? AND ?"
        total_count = conn.execute(total_count_query, (category['uuid'], start_month_str, end_month_str)).fetchone()[0]

        top_by_frequency.append({
            "name": category['name'],
            "value": category['value'],
            "details": {"items": details, "total_count": total_count}
        })

    conn.close()
    
    return {
        "by_amount": top_by_amount,
        "by_frequency": top_by_frequency
    }

def get_all_budgets():
    """모든 예산 설정을 조회 (지출액 집계 없음)"""
    conn = database.get_db_connection()
    query = """--sql
        SELECT
            b.id,
            b.budget_type,
            b.target_id,
            b.amount,
            CASE
                WHEN b.budget_type = 'major' THEN mc.name
                WHEN b.budget_type = 'minor' THEN mnc.name
            END as target_name,
            COALESCE(mnc.major_category_id, CAST(b.target_id AS INTEGER)) as major_category_id,
            CASE
                WHEN b.budget_type = 'major' THEN mc.name
                WHEN b.budget_type = 'minor' THEN mc_of_mnc.name
            END as major_category_name
        FROM budgets b
        LEFT JOIN major_categories mc ON b.budget_type = 'major' AND b.target_id = CAST(mc.id AS TEXT)
        LEFT JOIN minor_categories mnc ON b.budget_type = 'minor' AND b.target_id = mnc.uuid
        LEFT JOIN major_categories mc_of_mnc ON mnc.major_category_id = mc_of_mnc.id
    """
    budgets = [dict(row) for row in conn.execute(query).fetchall()]
    conn.close()
    return budgets

def add_budget(data):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO budgets (budget_type, target_id, amount) VALUES (?, ?, ?)",
        (data['budget_type'], data['target_id'], data['amount'])
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return {'id': new_id, **data}

def update_budget(budget_id, data):
    conn = database.get_db_connection()
    conn.execute(
        "UPDATE budgets SET budget_type = ?, target_id = ?, amount = ? WHERE id = ?",
        (data['budget_type'], data['target_id'], data['amount'], budget_id)
    )
    conn.commit()
    conn.close()
    return {'id': budget_id, **data}

def delete_budget(budget_id):
    conn = database.get_db_connection()
    conn.execute("DELETE FROM budgets WHERE id = ?", (budget_id,))
    conn.commit()
    conn.close()
    return {'message': 'Budget deleted successfully'}