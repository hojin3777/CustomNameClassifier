import React, { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, Label, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Brush, Legend } from 'recharts';
import { PiChartDonutFill } from 'react-icons/pi';
import { AiOutlineAreaChart } from 'react-icons/ai';
import './AssetPortfolio.css';

const API_BASE_URL = 'http://localhost:5000';

// ****** 타입 정의 ******
interface AccountBalance {
    account_name: string;
    balance: number;
}

interface MonthlyAccountData {
    month: string;
    accounts: {
        account_id: number;
        account_name: string;
        balance: number;
    }[];
    total: number;
}

interface AssetPortfolioProps {
    // props 없음 (전체 기간 독립적으로 동작)
}

// ****** 메인 컴포넌트 ******
const AssetPortfolio: React.FC<AssetPortfolioProps> = () => {
    const [monthlyData, setMonthlyData] = useState<MonthlyAccountData[]>([]);
    const [loading, setLoading] = useState(true);

    // ****** 데이터 로딩 ******
    useEffect(() => {
        loadAssetData();
    }, []);

    const loadAssetData = async () => {
        setLoading(true);
        try {
            const monthlyRes = await fetch(`${API_BASE_URL}/api/statistics/asset_portfolio_monthly`);
            const monthlyDataRaw = await monthlyRes.json();
            setMonthlyData(monthlyDataRaw);
        } catch (error) {
            console.error('Error loading asset data:', error);
        } finally {
            setLoading(false);
        }
    };

    // ****** 영역 차트 데이터 변환 ******
    const areaChartData = monthlyData.map(monthData => {
        const dataPoint: any = { month: monthData.month };
        monthData.accounts.forEach(acc => {
            dataPoint[acc.account_name] = acc.balance;
        });
        dataPoint.total = monthData.total;
        return dataPoint;
    });

    // 모든 계좌명 목록 (범례용)
    const allAccountNames = monthlyData.length > 0
        ? monthlyData[0].accounts.map(acc => acc.account_name)
        : [];

    // ****** 색상 팔레트 ******
    const COLORS = [
        'var(--color-highlight-2-transparent9)',
        'var(--color-highlight-3-transparent9)',
        'var(--color-highlight-4-transparent9)',
        'var(--color-highlight-5-transparent9)',
        'var(--color-highlight-6-transparent9)',
        'var(--color-highlight-1-transparent9)',
        'var(--color-highlight-2-transparent7)',
        'var(--color-highlight-3-transparent7)',
        'var(--color-highlight-4-transparent7)',
        'var(--color-highlight-5-transparent7)',
        'var(--color-highlight-6-transparent7)',
        'var(--color-highlight-1-transparent7)'
    ];

    // ****** 커스텀 툴팁 (영역 차트용) ******
    const CustomTooltip = ({ active, payload }: any) => {
        if (!active || !payload || payload.length === 0) return null;

        const month = payload[0].payload.month;
        const total = payload[0].payload.total;

        // 전월 대비 증감 계산
        const currentIndex = areaChartData.findIndex(d => d.month === month);
        let changeAmount = 0;
        let changePercent = 0;

        if (currentIndex > 0) {
            const prevTotal = areaChartData[currentIndex - 1].total;
            changeAmount = total - prevTotal;
            changePercent = prevTotal !== 0 ? (changeAmount / prevTotal) * 100 : 0;
        }

        return (
            <div className="asset-tooltip-aportfolio">
                <div className="tooltip-header-aportfolio">{month}</div>
                <div className="tooltip-total-aportfolio">
                    총 자산: {total.toLocaleString()}원
                </div>
                {currentIndex > 0 && (
                    <div className={`tooltip-change-aportfolio ${changeAmount >= 0 ? 'positive' : 'negative'}`}>
                        전월 대비: {changeAmount >= 0 ? '+' : ''}{changeAmount.toLocaleString()}원 ({changePercent.toFixed(1)}%)
                    </div>
                )}
                <div className="tooltip-divider-aportfolio"></div>
                {payload.map((entry: any, index: number) => {
                    if (entry.dataKey === 'total') return null;
                    return (
                        <div key={index} className="tooltip-item-aportfolio">
                            <span style={{ color: entry.color }}>{entry.name}:</span>
                            <span>{entry.value.toLocaleString()}원</span>
                        </div>
                    );
                })}
            </div>
        );
    };

    // ****** 이중 X축 렌더링 함수 ******
    // 월 표시 (숫자만)
    const monthTickFormatter = (tick: string) => {
        return tick.substring(2, 4) + '-' + tick.substring(5); // "2024-08" → "24-08"
        // return tick.substring(5); // "2024-08" → "08"
    };
    const renderMonthTick = (props: any) => {
        const { x, y, payload } = props;
        const tick = payload.value;
        const month = tick.substring(5); // "MM"
        const year = tick.substring(2, 4); // "YY"

        return (
            <g transform={`translate(${x},${y})`}>
                <text
                    x={0}
                    y={0}
                    dy={8}
                    textAnchor="middle"
                    fill="var(--color-text-secondary)"
                    fontSize={12}
                >
                    <tspan x={1} dy={10}>{year}</tspan>  {/* 첫 번째 줄: YY */}
                    <tspan x={0} dy={18}>{month}</tspan>  {/* 두 번째 줄: MM */}
                </text>
                <line x1={0} y1={11} x2={0} y2={18} stroke="var(--color-text-secondary)" strokeWidth={0.5} />
            </g>
        );
    };

    const brushTickFormatter = (tick: string, index: number) => {
        const month = tick.substring(5); // "MM"
        const year = tick.substring(2, 4); // "YY"

        return `${year}-${month}`;
    };


    // ****** 렌더링 ******
    if (loading) {
        return (
            <div className="dashboard-card">
                <div className="dashboard-card-header">
                    <h3 className="dashboard-card-title">자산 포트폴리오</h3>
                </div>
                <div className="dashboard-card-content">
                    <div className="loading-message-aportfolio">데이터 로딩 중...</div>
                </div>
            </div>
        );
    }

    return (
        <div className="dashboard-card">
            <div className="dashboard-card-header">
                <h3 className="dashboard-card-title">자산 포트폴리오</h3>
                <div className="dashboard-card-subtitle">전체 기간 기준</div>
            </div>

            <div className="dashboard-card-content asset-content-aportfolio">
                <div className="area-tab-container-aportfolio">
                    <ResponsiveContainer width="100%" height={470}>
                        <AreaChart data={areaChartData} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" />
                            <XAxis
                                dataKey="month"
                                stroke="var(--color-text-secondary)"
                                // tick={{ fill: 'var(--color-text-secondary)', fontSize: 12 }}
                                // tickFormatter={monthTickFormatter}
                                // height={30}
                                tick={renderMonthTick}
                                height={40}
                            />
                            <YAxis
                                stroke="var(--color-text-secondary)"
                                tick={{ fill: 'var(--color-text-secondary)', fontSize: 12 }}
                                tickCount={7}
                                tickFormatter={(value) => `${(value / 10000).toFixed(0)}만`}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend
                                wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }}
                                iconType="rect"
                            />
                            {allAccountNames.map((accName, index) => (
                                <Area
                                    key={accName}
                                    type="monotone"
                                    dataKey={accName}
                                    stackId="1"
                                    stroke={COLORS[index % COLORS.length]}
                                    fill={COLORS[index % COLORS.length]}
                                    isAnimationActive={true}
                                />
                            ))}
                            <Brush
                                dataKey="month"
                                height={30}
                                stroke="var(--color-accent-blue)"
                                fill="var(--color-bg-content)"
                                tickFormatter={brushTickFormatter}
                                y={400}
                                startIndex={0}
                                endIndex={areaChartData.length - 1}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};

export default AssetPortfolio;