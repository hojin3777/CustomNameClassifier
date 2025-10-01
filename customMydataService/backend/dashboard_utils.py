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

def get_type_ratio(year, month):
    """지정된 월의 수입/지출 유형별 비율을 계산합니다."""
    conn = database.get_db_connection()
    query = """
        SELECT
            type,
            SUM(ABS(amount)) as value
        FROM transactions
        WHERE strftime('%Y', transaction_date) = ? AND strftime('%m', transaction_date) = ?
        AND type IS NOT NULL AND type != ''
        GROUP BY type
    """
    
    cursor = conn.execute(query, (str(year), f'{month:02d}'))
    rows = cursor.fetchall()
    conn.close()

    income_types = ['고정수입', '유동수입']
    expense_types = ['고정지출', '반고정지출', '유동지출']
    
    income_data = [{'name': row['type'], 'value': row['value']} for row in rows if row['type'] in income_types]
    expense_data = [{'name': row['type'], 'value': row['value']} for row in rows if row['type'] in expense_types]

    return {'income': income_data, 'expense': expense_data}

def get_category_spending(year, month):
    """지정된 월의 대분류별 지출을 계산합니다."""
    conn = database.get_db_connection()
    query = """
        SELECT
            mc.name,
            SUM(ABS(t.amount)) as value
        FROM transactions t
        JOIN minor_categories mnc ON t.minor_category_uuid = mnc.uuid
        JOIN major_categories mc ON mnc.major_category_id = mc.id
        WHERE strftime('%Y', t.transaction_date) = ? 
          AND strftime('%m', t.transaction_date) = ?
          AND t.amount < 0
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