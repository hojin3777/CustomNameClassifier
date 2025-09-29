import database
from datetime import datetime

def get_monthly_summary(start_date_str, end_date_str):
    """
    지정된 기간 동안의 월별 총 수입과 총 지출을 계산합니다.
    """
    conn = database.get_db_connection()
    
    # 날짜 형식 유효성 검사 및 변환
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        # 날짜 형식이 잘못되었을 경우 빈 데이터를 반환
        return []

    query = """
        SELECT
            strftime('%Y-%m', transaction_date) AS month,
            SUM(CASE WHEN type = '수입' THEN amount ELSE 0 END) AS total_income,
            -- '지출'로 끝나는 모든 유형을 합산하도록 수정
            SUM(ABS(CASE WHEN type LIKE '%지출' THEN amount ELSE 0 END)) AS total_expense
        FROM transactions
        -- WHERE 조건도 '지출'로 끝나는 모든 유형을 포함하도록 수정
        WHERE transaction_date BETWEEN ? AND ?
        AND (type = '수입' OR type LIKE '%지출')
        GROUP BY month
        ORDER BY month ASC
    """
    
    cursor = conn.execute(query, (start_date, end_date))
    summary = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return summary