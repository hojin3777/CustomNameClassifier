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
    const [activeTab, setActiveTab] = useState<'donut' | 'area'>('donut');
    const [currentBalances, setCurrentBalances] = useState<AccountBalance[]>([]);
    const [monthlyData, setMonthlyData] = useState<MonthlyAccountData[]>([]);
    const [loading, setLoading] = useState(true);

    // ****** 데이터 로딩 ******
    useEffect(() => {
        loadAssetData();
    }, []);

    const loadAssetData = async () => {
        setLoading(true);
        try {
            // 1. 현재 잔액 조회 (도넛 차트용)
            const balancesRes = await fetch(`${API_BASE_URL}/api/statistics/account_balances`);
            const balancesData = await balancesRes.json();
            setCurrentBalances(balancesData);

            // 2. 월별 누적 잔액 조회 (영역 차트용)
            const monthlyRes = await fetch(`${API_BASE_URL}/api/statistics/asset_portfolio_monthly`);
            const monthlyDataRaw = await monthlyRes.json();
            setMonthlyData(monthlyDataRaw);
        } catch (error) {
            console.error('Error loading asset data:', error);
        } finally {
            setLoading(false);
        }
    };

    // ****** 도넛 차트 데이터 변환 ******
    const donutChartData = currentBalances.map(acc => ({
        name: acc.account_name,
        value: acc.balance
    }));

    // ****** 도넛 차트 커스텀 툴팁 ******
    const CustomDonutTooltip = ({ active, payload }: any) => {
        if (!active || !payload || payload.length === 0) return null;

        const data = payload[0];
        const percentage = totalBalance !== 0 ? (data.value / totalBalance) * 100 : 0;

        return (
            <div className="asset-tooltip-aportfolio">
                <div className="tooltip-header-aportfolio">{data.name}</div>
                <div className="tooltip-total-aportfolio">
                    {data.value.toLocaleString()}원
                </div>
                <div style={{ color: 'var(--color-text-secondary)', fontSize: '13px', marginTop: '4px' }}>
                    전체의 {percentage.toFixed(1)}%
                </div>
            </div>
        );
    };

    const totalBalance = donutChartData.reduce((sum, item) => sum + item.value, 0);

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

    // useEffect(() => {
    //     if (!chartContainerRef.current || areaChartData.length === 0) return;

    //     const measureTickInterval = () => {
    //         // Recharts가 렌더링한 X축 tick 요소들을 찾습니다
    //         const ticks = chartContainerRef.current?.querySelectorAll('.recharts-xAxis .recharts-cartesian-axis-tick');
    //         if (!ticks || ticks.length < 2) return;

    //         // 첫 번째와 두 번째 tick의 x 좌표 차이를 계산
    //         const firstTick = ticks[0] as SVGGElement;
    //         const secondTick = ticks[1] as SVGGElement;

    //         const firstX = firstTick.getAttribute('transform')?.match(/translate\(([^,]+)/)?.[1];
    //         const secondX = secondTick.getAttribute('transform')?.match(/translate\(([^,]+)/)?.[1];

    //         if (firstX && secondX) {
    //             const interval = parseFloat(secondX) - parseFloat(firstX);
    //             setActualTickInterval(interval);
    //         }
    //     };

    //     // 차트가 렌더링된 후 측정 (약간의 지연 필요)
    //     const timer = setTimeout(measureTickInterval, 100);

    //     // ResizeObserver로 차트 크기 변경 감지
    //     const observer = new ResizeObserver(() => {
    //         measureTickInterval();
    //     });

    //     if (chartContainerRef.current) {
    //         observer.observe(chartContainerRef.current);
    //     }

    //     return () => {
    //         clearTimeout(timer);
    //         observer.disconnect();
    //     };
    // }, [areaChartData, activeTab]); // activeTab 변경 시에도 재측정

    // ****** 이중 X축 렌더링 함수 ******
    // 월 표시 (숫자만)
    const monthTickFormatter = (tick: string) => {
        return tick.substring(2, 4) + '-' + tick.substring(5); // "2024-08" → "24-08"
        // return tick.substring(5); // "2024-08" → "08"
    };

    // const renderYearTick = (tickProps: any) => {
    //     const { x, y, payload, index } = tickProps;
    //     const { value } = payload;
    //     const currentMonth = value.substring(5); // "MM"
    //     const currentYear = value.substring(0, 4); // "YYYY"

    //     // 해당 년도의 시작/끝 인덱스 찾기
    //     let yearStartIndex = -1;
    //     let yearEndIndex = -1;

    //     for (let i = 0; i < areaChartData.length; i++) {
    //         const dataYear = areaChartData[i].month.substring(0, 4);
    //         if (dataYear === currentYear) {
    //             if (yearStartIndex === -1) yearStartIndex = i; // 첫 등장
    //             yearEndIndex = i; // 마지막 등장 계속 업데이트
    //         }
    //     }
    //     // console.log('Year Indices:', currentYear, yearStartIndex, yearEndIndex);

    //     const middleIndex = (yearStartIndex + yearEndIndex) / 2; // 중간 인덱스
    //     const middleX = x + ((middleIndex - index) * actualTickInterval); // 중간 위치 x 좌표
    //     const showDivider = index > 0; // 년도 경계 구분선 (첫 년도가 아닐 때만)

    //     // 첫 데이터이거나 마지막 데이터면 끝 구분선 표시
    //     if (index === 0 || index === areaChartData.length - 1) {
    //         return (
    //             <g>
    //                 {index === 0 && ( // 그중에서도 첫번째 년도 표시
    //                     <text x={middleX} y={y - 4} textAnchor="middle" fill="var(--color-text-secondary)" fontSize={11}>
    //                         {currentYear}
    //                     </text>
    //                 )}
    //                 <line x1={x} y1={y - 6} x2={x} y2={y - 35} stroke="var(--color-border-subtle)" strokeWidth={1} />
    //             </g>
    //         );
    //     }
    //     // 현재 데이터가 해당 년도의 첫 번째가 아니면 년도 label 표시 안함 (중복 방지)
    //     if (index !== yearStartIndex) {
    //         return null;
    //     }
    //     // 년도 라벨과 구분선 렌더링
    //     return (
    //         <g>
    //             {/* 구분선 */}
    //             {showDivider && (
    //                 <line
    //                     x1={x - actualTickInterval / 2}
    //                     y1={y - 4}
    //                     x2={x - actualTickInterval / 2}
    //                     y2={y - 35}
    //                     stroke="var(--color-border-subtle)"
    //                     strokeWidth={1}
    //                 />
    //             )}
    //             {/* 년도 텍스트 */}
    //             <text
    //                 x={middleX}
    //                 y={y - 4}
    //                 textAnchor="middle"
    //                 fill="var(--color-text-secondary)"
    //                 fontSize={11}
    //             >
    //                 {currentYear}
    //             </text>
    //         </g>
    //     );
    // };

    const brushTickFormatter = (tick: string, index: number) => {
        const month = tick.substring(5); // "MM"
        const year = tick.substring(2, 4); // "YY"

        // 1월이거나 첫 데이터면 년도 표시
        if (month === '01' || index === 0) {
            return `${year}-${month}`;
        }

        // 나머지는 월만 표시
        return month;
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
                <div className="header-left-aportfolio">
                    <h3 className="dashboard-card-title">자산 포트폴리오</h3>
                    {/* 탭 버튼 */}
                    <div className="tab-buttons-aportfolio">
                        <button
                            className={`tab-button-aportfolio ${activeTab === 'donut' ? 'active' : ''}`}
                            onClick={() => setActiveTab('donut')}
                            title="계좌별 잔액"
                        >
                            <PiChartDonutFill />
                        </button>
                        <button
                            className={`tab-button-aportfolio ${activeTab === 'area' ? 'active' : ''}`}
                            onClick={() => setActiveTab('area')}
                            title="자산 추이"
                        >
                            <AiOutlineAreaChart />
                        </button>
                    </div>
                </div>
                <div className="dashboard-card-subtitle">전체 기간 기준</div>
            </div>

            <div className="dashboard-card-content asset-content-aportfolio">
                {activeTab === 'donut' ? (
                    // ****** 도넛 차트 탭 ******
                    <div className="donut-tab-container-aportfolio">
                        <div className="donut-chart-wrapper-aportfolio">
                            <ResponsiveContainer width="100%" height={400}>
                                <PieChart>
                                    <Pie
                                        data={donutChartData}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={80}
                                        outerRadius={140}
                                        dataKey="value"
                                        isAnimationActive={true}
                                    >
                                        {donutChartData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                        ))}
                                        {/* 중앙 총액 Label */}
                                        <Label
                                            value={`총 ${totalBalance.toLocaleString()}원`}
                                            position="center"
                                            style={{
                                                fontSize: '16px',
                                                fontWeight: 'bold',
                                                fill: 'var(--color-text-primary)'
                                            }}
                                        />
                                    </Pie>
                                    <Tooltip content={<CustomDonutTooltip />} />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>

                        {/* 계좌별 잔액 테이블 */}
                        <div className="balance-table-wrapper-aportfolio">
                            <table className="balance-table-aportfolio">
                                <thead>
                                    <tr>
                                        <th>계좌명</th>
                                        <th>잔액</th>
                                        <th>비율</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {currentBalances.map((acc, index) => (
                                        <tr key={index}>
                                            <td className="account-name-cell-aportfolio">
                                                <span
                                                    className="color-dot-aportfolio"
                                                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                                                ></span>
                                                {acc.account_name}
                                            </td>
                                            <td className="balance-cell-aportfolio">{acc.balance.toLocaleString()}원</td>
                                            <td className="ratio-cell-aportfolio">
                                                {totalBalance !== 0 ? ((acc.balance / totalBalance) * 100).toFixed(1) : 0}%
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                ) : (
                    // ****** 영역 차트 탭 ******
                    <div className="area-tab-container-aportfolio">
                        <ResponsiveContainer width="100%" height={470}>
                            <AreaChart data={areaChartData} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" />
                                <XAxis
                                    dataKey="month"
                                    // tickFormatter={monthTickFormatter}
                                    stroke="var(--color-text-secondary)"
                                    tick={{ fill: 'var(--color-text-secondary)', fontSize: 12 }}
                                    height={30}
                                />
                                {/* <XAxis
                                    dataKey="month"
                                    axisLine={false}
                                    tickLine={false}
                                    tick={renderYearTick as any}
                                    interval={0}
                                    height={1}
                                    xAxisId="year"
                                    allowDataOverflow={true}
                                /> */}
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
                )}
            </div>
        </div>
    );
};

export default AssetPortfolio;