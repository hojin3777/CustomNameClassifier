import { useState, useEffect } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid, LabelList } from 'recharts';
import './MonthlyDetail.css';

// --- 타입 정의 (변경 없음) ---
interface MonthlySummary {
  fixed_income: number;
  variable_income: number;
  fixed_expense: number;
  semi_fixed_expense: number;
  variable_expense: number;
}
interface CategorySpending {
  name: string;
  value: number;
  percentage: number;
}
interface MonthlyDetailProps {
  selectedYear: number | null;
  selectedMonth: number | null;
}

// --- 범례 컴포넌트 (월별 추이와 동일) ---
const CustomLegend = (props: any) => {
  const { payload, chartData } = props;
  if (!chartData || chartData.length === 0) return null;
  const data = chartData[0];

  const filteredPayload = payload.filter((entry: any) => {
    if (entry.value === '초과지출') return data.deficit > 0;
    if (entry.value === '잉여금') return data.surplus > 0;
    // 0원인 항목은 범례에서 제외
    if (data[entry.dataKey] === 0) return false;
    return true;
  });

  return (
    <ul className="custom-legend-mdetail">
      {filteredPayload.map((entry: any, index: number) => (
        <li key={`item-${index}`} style={{ color: entry.color }}>
          <span className="legend-icon-mdetail" style={{ backgroundColor: entry.color }}></span>
          {entry.value}
        </li>
      ))}
    </ul>
  );
};

const CustomTooltip = ({ active, payload, selectedYear, selectedMonth }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const totalIncome = data.fixed_income + data.variable_income;
    const totalExpense = data.fixed_expense + data.semi_fixed_expense + data.variable_expense;
    const difference = totalIncome - totalExpense;

    // 각 항목의 색상을 payload에서 추출하여 맵으로 만듭니다.
    const colorMap = payload.reduce((acc: any, p: any) => {
      acc[p.dataKey] = p.fill;
      return acc;
    }, {});

    return (
      // 4. 툴팁 배경/테두리 및 고유 클래스명 적용
      <div className="custom-tooltip-mdetail">
        <p className="tooltip-label-mdetail">{`${selectedYear}년 ${selectedMonth}월 상세`}</p>
        <div className="tooltip-section-mdetail">
          <p className="tooltip-item-mdetail income">
            <span>총수입</span><span>{totalIncome.toLocaleString()}원</span>
          </p>
          <p className="tooltip-item-mdetail sub-item">
            <span style={{ color: colorMap.fixed_income }}>&nbsp;&nbsp;├ 고정수입</span><span>{data.fixed_income.toLocaleString()}원</span>
          </p>
          <p className="tooltip-item-mdetail sub-item">
            <span style={{ color: colorMap.variable_income }}>&nbsp;&nbsp;└ 유동수입</span><span>{data.variable_income.toLocaleString()}원</span>
          </p>
        </div>
        <div className="tooltip-section-mdetail">
          <p className="tooltip-item-mdetail expense">
            <span>총지출</span><span>{totalExpense.toLocaleString()}원</span>
          </p>
          <p className="tooltip-item-mdetail sub-item">
            <span style={{ color: colorMap.fixed_expense }}>&nbsp;&nbsp;├ 고정지출</span><span>{data.fixed_expense.toLocaleString()}원</span>
          </p>
          <p className="tooltip-item-mdetail sub-item">
            <span style={{ color: colorMap.semi_fixed_expense }}>&nbsp;&nbsp;├ 반고정지출</span><span>{data.semi_fixed_expense.toLocaleString()}원</span>
          </p>
          <p className="tooltip-item-mdetail sub-item">
            <span style={{ color: colorMap.variable_expense }}>&nbsp;&nbsp;└ 유동지출</span><span>{data.variable_expense.toLocaleString()}원</span>
          </p>
        </div>
        <div className="tooltip-section-mdetail">
          <p className={`tooltip-item-mdetail ${difference >= 0 ? 'surplus' : 'deficit'}`}>
            <span>차액</span><span>{difference.toLocaleString()}원</span>
          </p>
        </div>
      </div>
    );
  }
  return null;
};

const MonthlyDetail = ({ selectedYear, selectedMonth }: MonthlyDetailProps) => {
  const [monthlySummary, setMonthlySummary] = useState<MonthlySummary | null>(null);
  const [categorySpending, setCategorySpending] = useState<CategorySpending[]>([]);

  // --- useEffect 데이터 호출 (변경 없음) ---
  useEffect(() => {
    const fetchMonthlyDetails = async () => {
      if (!selectedYear || !selectedMonth) {
        setMonthlySummary(null);
        setCategorySpending([]);
        return;
      }
      try {
        const summaryResponse = await fetch(
          `http://localhost:5000/api/statistics/monthly_detail?year=${selectedYear}&month=${selectedMonth}`
        );
        if (summaryResponse.ok) setMonthlySummary(await summaryResponse.json());
        else setMonthlySummary(null);

        const categoryResponse = await fetch(
          `http://localhost:5000/api/statistics/category_spending?year=${selectedYear}&month=${selectedMonth}`
        );
        if (categoryResponse.ok) setCategorySpending((await categoryResponse.json()) || []);
        else setCategorySpending([]);
      } catch (error) {
        console.error('Error fetching monthly details:', error);
        setMonthlySummary(null);
        setCategorySpending([]);
      }
    };
    fetchMonthlyDetails();
  }, [selectedYear, selectedMonth]);


  // --- 차트 데이터 및 차액 계산 (월별 추이와 동일한 로직) ---
  const totalIncome = monthlySummary ? monthlySummary.fixed_income + monthlySummary.variable_income : 0;
  const totalExpense = monthlySummary
    ? monthlySummary.fixed_expense + monthlySummary.semi_fixed_expense + monthlySummary.variable_expense
    : 0;
  const difference = totalIncome - totalExpense;

  const chartData = [
    {
      name: '유형', // BarChart는 카테고리 축을 위한 name이 필요합니다.
      ...monthlySummary,
      surplus: difference > 0 ? difference : 0,
      deficit: difference < 0 ? -difference : 0,
    },
  ];

  // Y축 눈금을 깔끔하게 만들기 위한 도메인 계산
  const rawMax = Math.max(totalIncome, totalExpense);
  const maxDomain = rawMax > 0 ? Math.ceil(rawMax / 100000) * 100000 : 100000; // 10만 단위로 올림
  const formatYAxis = (tick: number) => `${tick / 10000}`;
  const formatLabel = (value: any) => {
    if (typeof value !== 'number' || value === 0) {
      return '';
    }
    return (value / 10000).toLocaleString(undefined, {
      minimumFractionDigits: 0,
      maximumFractionDigits: 1,
    });
  };

  return (
    <div className="dashboard-card">
      <div className="dashboard-card-header">
        <h3 className="dashboard-card-title">월간 상세 분석<span className='dashboard-card-title subtle'> (단위:만)</span></h3>
        <div className="dashboard-card-subtitle">
          {selectedYear && selectedMonth ? `${selectedYear}년 ${selectedMonth}월 기준` : '데이터를 선택하세요'}
        </div>
      </div>
      <div className="dashboard-card-content monthly-details-grid">
        <div className='chart-area-wrapper'>
          <div className="chart-container-single">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }} barCategoryGap="35%">
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border-subtle)" />
                <XAxis type="category" dataKey="name" tickLine={false} axisLine={false} tick={false} />
                <YAxis type="number" domain={[0, maxDomain]} tickFormatter={formatYAxis} tick={{ fill: 'var(--color-text-secondary)' }} tickCount={8} />
                <Tooltip content={<CustomTooltip selectedYear={selectedYear} selectedMonth={selectedMonth} />} cursor={{ fill: 'var(--color-bg-overlay-light)' }} />

                {/* 수입 막대 그룹 */}
                <Bar dataKey="fixed_income" name="고정수입" stackId="income" fill="var(--color-highlight-4)" barSize={50}>
                  <LabelList dataKey="fixed_income" position="center" formatter={formatLabel} fill="#fff" fontSize={12} />
                </Bar>
                <Bar dataKey="variable_income" name="유동수입" stackId="income" fill="var(--color-highlight-5)" barSize={50}>
                  <LabelList dataKey="variable_income" position="center" formatter={formatLabel} fill="#fff" fontSize={12} />
                </Bar>
                <Bar dataKey="deficit" name="초과지출" stackId="income" fill="var(--color-highlight-1-transparent5)" barSize={50} />

                {/* 지출 막대 그룹 */}
                <Bar dataKey="fixed_expense" name="고정지출" stackId="expense" fill="var(--color-highlight-2)" barSize={50}>
                  <LabelList dataKey="fixed_expense" position="center" formatter={formatLabel} fill="#fff" fontSize={12} />
                </Bar>
                <Bar dataKey="semi_fixed_expense" name="반고정지출" stackId="expense" fill="var(--color-highlight-6)" barSize={50}>
                  <LabelList dataKey="semi_fixed_expense" position="center" formatter={formatLabel} fill="#fff" fontSize={12} />
                </Bar>
                <Bar dataKey="variable_expense" name="유동지출" stackId="expense" fill="var(--color-highlight-3)" barSize={50}>
                  <LabelList dataKey="variable_expense" position="center" formatter={formatLabel} fill="#fff" fontSize={12} />
                </Bar>
                <Bar dataKey="surplus" name="잉여금" stackId="expense" fill="var(--color-highlight-1-transparent5)" barSize={50} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <CustomLegend payload={[
            { value: '고정수입', color: 'var(--color-highlight-4)', dataKey: 'fixed_income' },
            { value: '유동수입', color: 'var(--color-highlight-5)', dataKey: 'variable_income' },
            { value: '고정지출', color: 'var(--color-highlight-2)', dataKey: 'fixed_expense' },
            { value: '반고정지출', color: 'var(--color-highlight-6)', dataKey: 'semi_fixed_expense' },
            { value: '유동지출', color: 'var(--color-highlight-3)', dataKey: 'variable_expense' },
            { value: '잉여금', color: 'var(--color-highlight-1-transparent5)', dataKey: 'surplus' },
            { value: '초과지출', color: 'var(--color-highlight-1-transparent5)', dataKey: 'deficit' },
          ]} chartData={chartData} />
        </div>


        {/* 2. 우측: 3열 테이블 영역 */}
        <div className="table-container">
          <h4>대분류별 지출</h4>
          <div className="category-table-wrapper">
            <table className="category-table">
              <thead>
                <tr>
                  <th>카테고리</th>
                  <th>금액</th>
                  <th>비율</th>
                </tr>
              </thead>
              <tbody>
                {categorySpending.length > 0 ? (
                  categorySpending.map((item, index) => (
                    <tr key={index}>
                      <td>{item.name}</td>
                      <td className="td-amount">{item.value.toLocaleString()}원</td>
                      <td className="td-percentage">
                        <div className="percentage-bar-outer">
                          <div
                            className="percentage-bar-inner"
                            style={{ width: `${item.percentage}%`, backgroundColor: `var(--color-chart-${index + 1})` }}
                          ></div>
                        </div>
                        <span>{item.percentage.toFixed(1)}%</span>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={3} className="no-data-message">해당 월의 지출 내역이 없습니다.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MonthlyDetail;