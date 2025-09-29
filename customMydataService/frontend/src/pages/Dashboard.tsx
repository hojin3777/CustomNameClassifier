import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Bar,
} from 'recharts';
import './Dashboard.css';

// API 응답 데이터 타입 정의
interface MonthlySummary {
  month: string;
  total_income: number;
  total_expense: number;
}

const Dashboard = () => {
  const [summaryData, setSummaryData] = useState<MonthlySummary[]>([]);

  useEffect(() => {
    // TODO: 이 날짜 범위를 툴바의 날짜 필터와 연동해야 합니다.
    // 우선 올해를 기본값으로 설정합니다.
    const currentYear = new Date().getFullYear();
    const startDate = `${currentYear}-01-01`;
    const endDate = `${currentYear}-12-31`;

    const fetchSummaryData = async () => {
      try {
        const response = await fetch(`http://localhost:5000/api/statistics/monthly_summary?start_date=${startDate}&end_date=${endDate}`);
        if (!response.ok) {
          throw new Error('Failed to fetch summary data');
        }
        const data: MonthlySummary[] = await response.json();
        const formattedData = data.map(item => ({
          month: `${parseInt(item.month.split('-')[1], 10)}월`,
          // 값이 숫자가 아닐 수 있는 경우를 대비해 명시적으로 숫자로 변환
          total_income: Number(item.total_income || 0),
          total_expense: Number(item.total_expense || 0),
        }));
        setSummaryData(formattedData);
        // alert(JSON.stringify(formattedData));
      } catch (error) {
        console.error("Error fetching summary data:", error);
      }
    };

    fetchSummaryData();
  }, []); // 빈 배열을 전달하여 컴포넌트가 처음 마운트될 때 한 번만 실행

  return (
    <div className="dashboard-page">
      <header className="main-header">
        <h1>Dashboard</h1>
      </header>

      <div className="dashboard-toolbar">
        {/* TODO: 기간 선택 필터 (연/월 셀렉터) */}
        <span>기간 필터 영역</span>
      </div>

      <main className="dashboard-content">
        {/* Phase 1: 좌측 영역 */}
        <div className="dashboard-card">
          <h3 className="dashboard-card-title">월별 지출/수입 추이</h3>
          <div className="dashboard-card-content">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={summaryData}
                margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="month" tick={{ fill: 'var(--color-text-secondary)' }} />
                <YAxis tick={{ fill: 'var(--color-text-secondary)' }} tickFormatter={(value) => new Intl.NumberFormat('ko-KR').format(value as number)} />
                <Tooltip
                  cursor={{ fill: 'var(--color-bg-content-hover)' }}
                  contentStyle={{
                    backgroundColor: 'var(--color-bg-header)',
                    borderColor: 'var(--color-border)',
                    borderRadius: '8px'
                  }}
                />
                <Legend />
                <Bar dataKey="total_income" name="수입" fill="var(--color-accent-green)" />
                <Bar dataKey="total_expense" name="지출" fill="var(--color-accent-red)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Phase 1: 우측 영역 (상단) */}
        <div className="dashboard-card">
          <h3 className="dashboard-card-title">월간 상세 분석</h3>
          <div className="dashboard-card-content">
            <p>(지출/수입 유형, 대분류 비율, 계좌 잔고가 표시될 영역)</p>
          </div>
        </div>

        {/* Phase 2 영역 (일단 비워둠) */}
        <div className="dashboard-card">
          <h3 className="dashboard-card-title">소비 TOP 10</h3>
          <div className="dashboard-card-content">
            <p>(Phase 2에서 구현될 영역)</p>
          </div>
        </div>
        
        <div className="dashboard-card">
          <h3 className="dashboard-card-title">소비 습관 히트맵</h3>
          <div className="dashboard-card-content">
            <p>(Phase 2에서 구현될 영역)</p>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;