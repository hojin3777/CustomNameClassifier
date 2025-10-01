import { useState, useEffect } from 'react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';
import type { PieLabelRenderProps } from 'recharts';

interface TypeRatio {
  name: string;
  value: number;
  [key: string]: any;
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

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#AF19FF', '#FF1919'];

const MonthlyDetail = ({ selectedYear, selectedMonth }: MonthlyDetailProps) => {
  const [incomeRatio, setIncomeRatio] = useState<TypeRatio[]>([]);
  const [expenseRatio, setExpenseRatio] = useState<TypeRatio[]>([]);
  const [categorySpending, setCategorySpending] = useState<CategorySpending[]>([]);

  useEffect(() => {
    const fetchMonthlyDetails = async () => {
      if (!selectedYear || !selectedMonth) {
        setIncomeRatio([]);
        setExpenseRatio([]);
        setCategorySpending([]);
        return;
      }

      try {
        const ratioResponse = await fetch(
          `http://localhost:5000/api/statistics/type_ratio?year=${selectedYear}&month=${selectedMonth}`
        );
        if (ratioResponse.ok) {
          const data = await ratioResponse.json();
          setIncomeRatio(data.income || []);
          setExpenseRatio(data.expense || []);
        } else {
          setIncomeRatio([]);
          setExpenseRatio([]);
        }

        const categoryResponse = await fetch(
          `http://localhost:5000/api/statistics/category_spending?year=${selectedYear}&month=${selectedMonth}`
        );
        if (categoryResponse.ok) {
          const data = await categoryResponse.json();
          setCategorySpending(data || []);
        } else {
          setCategorySpending([]);
        }
      } catch (error) {
        console.error('Error fetching monthly details:', error);
        setIncomeRatio([]);
        setExpenseRatio([]);
        setCategorySpending([]);
      }
    };

    fetchMonthlyDetails();
  }, [selectedYear, selectedMonth]);

  const renderCustomizedLabel = (props: PieLabelRenderProps) => {
    const { cx, cy, midAngle, innerRadius, outerRadius, percent } = props as any;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * (Math.PI / 180));
    const y = cy + radius * Math.sin(-midAngle * (Math.PI / 180));
    if (percent === 0) return null;
    return (
      <text x={x} y={y} fill="white" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central">
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  return (
    <div className="dashboard-card">
      <div className="dashboard-card-header">
        <h3 className="dashboard-card-title">월간 상세 분석</h3>
        <div className="dashboard-card-subtitle">
          {selectedYear && selectedMonth ? `${selectedYear}년 ${selectedMonth}월 기준` : '데이터가 없습니다'}
        </div>
      </div>
      <div className="dashboard-card-content monthly-details">
        <div className="chart-section">
          <h4>지출 유형</h4>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={expenseRatio}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={80}
                fill="#8884d8"
                labelLine={false}
                label={renderCustomizedLabel}
              >
                {expenseRatio.map((_, index) => (
                  <Cell key={`cell-expense-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-section">
          <h4>수입 유형</h4>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={incomeRatio}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={80}
                fill="#82ca9d"
                labelLine={false}
                label={renderCustomizedLabel}
              >
                {incomeRatio.map((_, index) => (
                  <Cell key={`cell-income-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="table-section">
          <h4>대분류별 지출</h4>
          <div className="category-spending-table">
            <table>
              <thead>
                <tr>
                  <th>카테고리</th>
                  <th>금액</th>
                  <th>비율</th>
                </tr>
              </thead>
              <tbody>
                {categorySpending.map((item, index) => (
                  <tr key={index}>
                    <td>{item.name}</td>
                    <td>{new Intl.NumberFormat('ko-KR').format(item.value)}</td>
                    <td>{item.percentage.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MonthlyDetail;
