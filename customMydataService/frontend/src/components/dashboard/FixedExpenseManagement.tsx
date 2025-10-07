import React, { useState, useEffect, useCallback, useRef } from 'react';
import { FaCircle, FaCaretUp, FaCaretDown, FaMinus } from 'react-icons/fa';
import './FixedExpenseManagement.css';

const API_BASE_URL = 'http://localhost:5000';

// ******************** 타입 정의 ********************
interface FixedExpenseItem {
    merchant: string;
    category: string;
    major_category: string;
    minor_category: string;
    day_range: string;
    amount_range: string;
    amount_min: number;
    amount_max: number;
    trend: 'up' | 'down' | 'same' | 'none';
    recent_months: [boolean, boolean, boolean];
    transaction_details: { date: string; amount: number }[];
    total_count: number;
    avg_count_per_month: number;
}

interface FixedExpenseManagementProps {
    months: string[];
    range: [number, number];
    onPopupStateChange?: (isOpen: boolean) => void;
    isPopupOpen?: boolean;
}

interface TooltipState {
    isOpen: boolean;
    position: { top: number; left: number };
    content: React.ReactNode;
}

// ******************** 메인 컴포넌트 ********************
const FixedExpenseManagement: React.FC<FixedExpenseManagementProps> = ({ months, range, onPopupStateChange, isPopupOpen }) => {
    const [expenses, setExpenses] = useState<FixedExpenseItem[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [tooltip, setTooltip] = useState<TooltipState>({ isOpen: false, position: { top: 0, left: 0 }, content: null });
    const tooltipRef = useRef<HTMLDivElement>(null);
    const rootRef = useRef<HTMLDivElement>(null);
    const [detailPopup, setDetailPopup] = useState<{
        isOpen: boolean;
        item: FixedExpenseItem | null
        position: { top: number; left: number };
    }>({ isOpen: false, item: null, position: { top: 0, left: 0 } });

    // ****** 데이터 로딩 ******
    const fetchData = useCallback(async () => {
        if (months.length === 0 || range[0] < 0 || range[1] < 0) return;

        const startMonth = months[range[0]];
        const endMonth = months[range[1]];

        setIsLoading(true);
        try {
            const response = await fetch(
                `${API_BASE_URL}/api/statistics/fixed_expenses?start_month=${startMonth}&end_month=${endMonth}`
            );
            const data = await response.json();
            setExpenses(data);
        } catch (error) {
            console.error('고정비 데이터 로딩 실패:', error);
        } finally {
            setIsLoading(false);
        }
    }, [months, range]);

    useEffect(() => {
        fetchData();
    }, [fetchData]);

    // ****** 툴팁 핸들러 ******
    // 거래처명/카테고리 툴팁
    const showDetailPopup = (e: React.MouseEvent, item: FixedExpenseItem) => {
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        const viewportHeight = window.innerHeight;
        const popupHeight = 240; // 최대 높이

        // 아래쪽 공간이 부족하면 위쪽에 표시
        const spaceBelow = viewportHeight - rect.bottom - 30;
        const spaceAbove = rect.top;

        let top = rect.bottom + window.scrollY + 5;

        if (spaceBelow < popupHeight && spaceAbove > spaceBelow) {
            // 위쪽에 표시
            top = rect.top + window.scrollY - popupHeight - 5;
        }

        setDetailPopup({
            isOpen: true,
            item,
            position: {
                top: top,
                left: rect.left + window.scrollX
            }
        });
        onPopupStateChange?.(true);
    };

    const hideDetailPopup = () => {
        setDetailPopup({ isOpen: false, item: null, position: { top: 0, left: 0 } });
        onPopupStateChange?.(false);
    };

    // ****** Dashboard에서 팝업 닫힘 감지 ******
    useEffect(() => {
        if (isPopupOpen === false && detailPopup.isOpen) {
            // Dashboard에서 팝업을 닫았으므로 로컬 상태도 업데이트
            setDetailPopup({ isOpen: false, item: null, position: { top: 0, left: 0 } });
        }
    }, [isPopupOpen]);  // isPopupOpen 변경 감지

    // ****** 외부 클릭 감지만 유지 (로컬 처리) ******
    useEffect(() => {
        if (!detailPopup.isOpen) return;

        const handleOutsideClick = (e: MouseEvent) => {
            const popupElement = document.querySelector('.detail-popup-tooltip-fexpense');
            if (popupElement && !popupElement.contains(e.target as Node)) {
                hideDetailPopup();
            }
        };

        document.addEventListener('mousedown', handleOutsideClick);

        return () => {
            document.removeEventListener('mousedown', handleOutsideClick);
        };
    }, [detailPopup.isOpen]);

    // 증감 아이콘 툴팁
    const showTrendTooltip = (e: React.MouseEvent, trend: string) => {
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        let message = '';
        if (trend === 'up') message = '전월 대비 증가';
        else if (trend === 'down') message = '전월 대비 감소';
        else if (trend === 'same') message = '전월과 동일';

        const content = <div className="fixed-expense-small-tooltip-fexpense">{message}</div>;

        setTooltip({
            isOpen: true,
            position: { top: rect.bottom - rect.height - 20, left: rect.left + (rect.width / 2) - 40 },
            content
        });
    };

    // 월별 지출 현황 툴팽
    const showMonthStatusTooltip = (e: React.MouseEvent, monthIndex: number, hasTransaction: boolean) => {
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        const last3Months = months.slice(Math.max(0, range[1] - 2), range[1] + 1);
        const targetMonth = last3Months[monthIndex];

        let message = '';
        if (!targetMonth) {
            message = '데이터 부족';
        } else if (hasTransaction) {
            const [year, month] = targetMonth.split('-');
            message = `${year}년 ${parseInt(month, 10)}월 결제완료`;
        } else {
            const [year, month] = targetMonth.split('-');
            message = `${year}년 ${parseInt(month, 10)}월 미결제`;
        }

        const content = <div className="fixed-expense-small-tooltip-fexpense">{message}</div>;

        setTooltip({
            isOpen: true,
            position: { top: rect.bottom - rect.height - 25, left: rect.left + (rect.width / 2) - 60 },
            content
        });
    };

    const hideTooltip = () => {
        setTooltip({ isOpen: false, position: { top: 0, left: 0 }, content: null });
    };

    // ****** 총 고정비 계산 ******
    const totalFixedExpense = expenses.reduce((sum, item) => {
        // 마지막 달 기준 평균 금액 사용
        return sum + (item.amount_min + item.amount_max) / 2;
    }, 0);

    // ****** 렌더링 ******
    if (isLoading) {
        return (
            <div className="dashboard-card">
                <div className="dashboard-card-header">
                    <h3 className="dashboard-card-title">고정비 관리
                        <span className="dashboard-card-title subtle">('고정지출' 유형이 2개월 이상 지속)</span>
                    </h3>
                    <div className="dashboard-card-subtitle">
                        {months.length > 0 && range[0] >= 0 && range[1] >= 0
                            ? `${months[range[0]].replace('-', '년 ')}월 ~ ${months[range[1]].replace('-', '년 ')}월`  // 형식 통일
                            : '기간을 선택하세요'}
                    </div>
                </div>
                <div className="dashboard-card-content fixed-expense-card-content-fexpense">
                    <div className="loading-message-fexpense">데이터 로딩 중...</div>
                </div>
            </div>
        );
    }

    return (
        <div className="dashboard-card" ref={rootRef}>
            <div className="dashboard-card-header">
                <h3 className="dashboard-card-title">고정비 관리
                    <span className="dashboard-card-title subtle"> ('고정지출' 유형이 2개월 이상 지속)</span>
                </h3>
                <div className="dashboard-card-subtitle">
                    {months.length > 0 && range[0] >= 0 && range[1] >= 0
                        ? `${months[range[0]].replace('-', '년 ')}월 ~ ${months[range[1]].replace('-', '년 ')}월`  // 형식 통일
                        : '기간을 선택하세요'}
                </div>
            </div>
            <div className="dashboard-card-content fixed-expense-card-content-fexpense">
                <div className="fixed-expense-table-container-fexpense">
                    <table className="fixed-expense-table-fexpense">
                        <thead>
                            <tr>
                                <th className="col-merchant-fexpense">거래처명</th>
                                <th className="col-category-fexpense">카테고리</th>
                                <th className="col-day-fexpense">평균 출금일</th>
                                <th className="col-amount-fexpense">&nbsp;&nbsp;&nbsp;&nbsp;평균 지출액</th>
                                <th className="col-trend-fexpense"></th>
                                <th className="col-recent-fexpense">월별 지출 현황</th>
                            </tr>
                        </thead>
                        <tbody>
                            {expenses.map((item, index) => (
                                <tr key={index}>
                                    {/* 거래처명 */}
                                    <td
                                        className="merchant-cell-fexpense"
                                        onClick={(e) => showDetailPopup(e, item)}
                                    >
                                        {item.merchant}
                                    </td>

                                    {/* 카테고리 */}
                                    <td
                                        className="category-cell-fexpense"
                                        onClick={(e) => showDetailPopup(e, item)}
                                    >
                                        {item.category}
                                    </td>

                                    {/* 평균 출금일 */}
                                    <td className="day-cell-fexpense">{item.day_range}</td>
                                    {/* 평균 지출액 */}
                                    <td className="amount-cell-fexpense">{item.amount_range}</td>
                                    {/* 증감 */}
                                    <td className="trend-cell-fexpense">
                                        {item.trend === 'up' && (
                                            <FaCaretUp
                                                className="trend-icon-up-fexpense"
                                                onMouseEnter={(e) => showTrendTooltip(e, 'up')}
                                                onMouseLeave={hideTooltip}
                                            />
                                        )}
                                        {item.trend === 'down' && (
                                            <FaCaretDown
                                                className="trend-icon-down-fexpense"
                                                onMouseEnter={(e) => showTrendTooltip(e, 'down')}
                                                onMouseLeave={hideTooltip}
                                            />
                                        )}
                                        {item.trend === 'same' && (
                                            <FaMinus
                                                className="trend-icon-same-fexpense"
                                                onMouseEnter={(e) => showTrendTooltip(e, 'same')}
                                                onMouseLeave={hideTooltip}
                                            />
                                        )}
                                    </td>

                                    {/* 월별 지출 현황 */}
                                    <td className="recent-cell-fexpense">
                                        <div className="recent-months-fexpense">
                                            {item.recent_months.map((hasTransaction, monthIdx) => (
                                                <FaCircle
                                                    key={monthIdx}
                                                    className={`month-indicator-fexpense ${hasTransaction === null
                                                        ? 'indicator-insufficient-fexpense'
                                                        : hasTransaction
                                                            ? 'indicator-complete-fexpense'
                                                            : 'indicator-incomplete-fexpense'
                                                        }`}
                                                    onMouseEnter={(e) => showMonthStatusTooltip(e, monthIdx, hasTransaction)}
                                                    onMouseLeave={hideTooltip}
                                                />
                                            ))}
                                        </div>
                                    </td>
                                </tr>
                            ))}
                            {expenses.length > 0 && (
                                <tr className="total-row-fexpense">
                                    <td></td>
                                    <td></td>
                                    <td className="total-label-fexpense">총 고정비</td>
                                    <td className="total-amount-fexpense">{totalFixedExpense.toLocaleString()}원</td>
                                    <td></td>
                                    <td></td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* 툴팁 */}
            {tooltip.isOpen && (
                <div
                    ref={tooltipRef}
                    className="fixed-expense-tooltip-fexpense"
                    style={{ top: tooltip.position.top, left: tooltip.position.left, pointerEvents: 'none' }}
                >
                    {tooltip.content}
                </div>
            )}
            {/* 상세 팝업 (거래처명/카테고리 클릭 시) */}
            {detailPopup.isOpen && detailPopup.item && (
                <>
                    {/* 툴팁 스타일 팝업 */}
                    <div
                        className="detail-popup-tooltip-fexpense"
                        style={{
                            top: `${detailPopup.position.top}px`,
                            left: `${detailPopup.position.left}px`
                        }}
                    >
                        <div className="detail-popup-header-fexpense">
                            <div className="detail-popup-title-fexpense">
                                <span>{detailPopup.item.merchant}</span>
                                <span>{detailPopup.item.total_count}회 (월 {detailPopup.item.avg_count_per_month}회)</span>
                            </div>
                            <button className="detail-popup-close-fexpense" onClick={hideDetailPopup}>✕</button>
                        </div>
                        <div className="detail-popup-period-fexpense">
                            {months[range[0]].replace('-', '년 ')}월 ~ {months[range[1]].replace('-', '년 ')}월
                        </div>
                        <div className="detail-popup-divider-fexpense"></div>
                        <div className="detail-popup-body-fexpense">
                            {detailPopup.item.transaction_details.map((tx, idx) => (
                                <div key={idx} className="detail-popup-item-fexpense">
                                    <span>{tx.date}</span>
                                    <span>{tx.amount.toLocaleString()}원</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </>
            )}
        </div>
    );
};

export default FixedExpenseManagement;