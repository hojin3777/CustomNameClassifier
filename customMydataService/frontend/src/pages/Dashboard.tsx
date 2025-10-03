import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import MonthlyTrend from '../components/dashboard/MonthlyTrend';
import MonthlyDetail from '../components/dashboard/MonthlyDetail';
import ComingSoon from '../components/dashboard/ComingSoon';
import FloatingSelectPopup, { type FloatingSelectHandle } from '../components/FloatingSelectPopup';
import './Dashboard.css';

const API_BASE_URL = 'http://localhost:5000';

const Dashboard = () => {
  const [availableMonths, setAvailableMonths] = useState<string[]>([]);
  const [range, setRange] = useState<[number, number]>([0, 0]);
  const [selectedYear, setSelectedYear] = useState<number | null>(null);
  const [selectedMonth, setSelectedMonth] = useState<number | null>(null);
  const floatingSelectRef = useRef<FloatingSelectHandle | null>(null);
  const rangeSelectorRef = useRef<HTMLDivElement | null>(null);
  const rangePopupRef = useRef<HTMLDivElement | null>(null);
  const sliderContainerRef = useRef<HTMLDivElement | null>(null);
  const [isRangePopupOpen, setIsRangePopupOpen] = useState(false);
  const [rangePopupPos, setRangePopupPos] = useState<{ top: number; left: number; width: number }>({
    top: 0,
    left: 0,
    width: 0,
  });

  // 사용가능한 월 및 range 로드
  useEffect(() => {
    const fetchAvailableMonths = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/statistics/available_months`);
        const months: string[] = await response.json();
        setAvailableMonths(months);

        if (months.length > 0) {
          const settingResponse = await fetch(`${API_BASE_URL}/api/settings/dashboard_trend_range`);
          const savedRange: [string, string] | null = await settingResponse.json();
          const startIndex = savedRange ? months.indexOf(savedRange[0]) : -1;
          const endIndex = savedRange ? months.indexOf(savedRange[1]) : -1;

          if (startIndex !== -1 && endIndex !== -1) {
            setRange([startIndex, endIndex]);
          } else {
            const defaultEndIndex = months.length - 1;
            const defaultStartIndex = Math.max(0, defaultEndIndex - 11);
            setRange([defaultStartIndex, defaultEndIndex]);
          }

          const [yearStr, monthStr] = months[months.length - 1].split('-');
          setSelectedYear(parseInt(yearStr, 10));
          setSelectedMonth(parseInt(monthStr, 10));
        }
      } catch (error) {
        console.error('Error fetching available months:', error);
      }
    };

    fetchAvailableMonths();
  }, []);

  // 슬라이더 너비 계산
  const sliderMonthWidth = 42;
  const sliderBaseWidth = useMemo(() => {
    if (availableMonths.length <= 1) return 320;
    const intervals = availableMonths.length - 1;
    return Math.max(intervals * sliderMonthWidth, 320);
  }, [availableMonths.length]);

  useEffect(() => {
    if (!isRangePopupOpen || !sliderContainerRef.current || availableMonths.length <= 1) return;
    const container = sliderContainerRef.current;
    const rightHandleIndex = range[1];
    const totalIntervals = availableMonths.length - 1;
    const handlePosition = (rightHandleIndex / totalIntervals) * sliderBaseWidth;
    let targetScrollLeft = handlePosition - container.clientWidth / 2;
    const maxScrollLeft = container.scrollWidth - container.clientWidth;
    targetScrollLeft = Math.max(0, Math.min(targetScrollLeft, maxScrollLeft));
    container.scrollLeft = targetScrollLeft;
  }, [isRangePopupOpen, range, availableMonths, sliderBaseWidth]);

  // Marks 객체 생성
  const sliderMarks = useMemo(() => {
    if (!availableMonths.length) return {};

    return availableMonths.reduce((acc, month, index) => {
      const [year, monthNum] = month.split('-');
      const isFirstMonthOfYear = monthNum === '01';
      const isFirstOverall = index === 0;

      const isInRange = index >= range[0] && index <= range[1];
      const isEdge = index === range[0] || index === range[1];

      const monthLabelClasses = [
        'slider-mark-node__month-label',
        isInRange ? 'slider-mark-node__month-label--selected' : '',
        isEdge ? 'slider-mark-node__month-label--edge' : '',
      ]
        .filter(Boolean)
        .join(' ');

      acc[index] = (
        <div className="slider-mark-node">
          {/* 연도 표시 로직 */}
          {(isFirstMonthOfYear || isFirstOverall) && (
            <>
              {!isFirstOverall && <span className="slider-mark-node__year-line" />}
              <span className="slider-mark-node__year-label">{year}</span>
            </>
          )}
          {/* 월 표시 로직 */}
          <span className={monthLabelClasses}>{monthNum}</span>
        </div>
      );
      return acc;
    }, {} as { [key: number]: React.ReactNode });
  }, [availableMonths, range]);

  // 나머지 핸들러 및 계산 (대부분 기존과 동일)
  const availableYears = useMemo(() => {
    const yearSet = new Set(availableMonths.map(month => month.substring(0, 4)));
    return Array.from(yearSet).sort((a, b) => parseInt(b, 10) - parseInt(a, 10));
  }, [availableMonths]);

  const monthsForSelectedYear = useMemo(() => {
    if (!selectedYear) return [];
    return availableMonths
      .filter(month => month.startsWith(`${selectedYear}-`))
      .map(month => month.substring(5));
  }, [availableMonths, selectedYear]);

  const formatMonthLabel = (monthStr?: string) => {
    if (!monthStr) return '';
    const [year, month] = monthStr.split('-');
    return `${year}년 ${parseInt(month, 10)}월`;
  };

  const rangeLabel = useMemo(() => {
    if (availableMonths.length === 0) return '기간 없음';
    const [startIndex, endIndex] = range;
    const clampedStart = Math.max(0, Math.min(startIndex, availableMonths.length - 1));
    const clampedEnd = Math.max(clampedStart, Math.min(endIndex, availableMonths.length - 1));
    return `${formatMonthLabel(availableMonths[clampedStart])} - ${formatMonthLabel(availableMonths[clampedEnd])}`;
  }, [availableMonths, range]);

  const handleSliderChange = (value: number | number[]) => {
    if (Array.isArray(value)) {
      setRange(value as [number, number]);
    }
  };

  const handleSliderChangeComplete = useCallback((value: number | number[]) => {
    if (Array.isArray(value) && availableMonths.length > 0) {
      const startMonth = availableMonths[value[0]];
      const endMonth = availableMonths[value[1]];
      // 서버에 변경된 기간을 저장
      if (startMonth && endMonth) {
        fetch(`${API_BASE_URL}/api/settings/dashboard_trend_range`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ range: [startMonth, endMonth] }),
        }).catch(error => console.error('Failed to save slider range setting:', error));
      }
    }
  }, [availableMonths]);


  // 팝업 관련 로직
  const openFloatingSelect = (target: HTMLElement, type: 'year' | 'month') => {
    const rect = target.getBoundingClientRect();
    const position = { top: rect.bottom, left: rect.left, width: rect.width };

    if (type === 'year') {
      const options = availableYears.map(year => ({ value: year, label: `${year}년` }));
      const currentValue = selectedYear ? selectedYear.toString() : '';
      floatingSelectRef.current?.open(options, currentValue, position, value => {
        const newYear = parseInt(value, 10);
        setSelectedYear(newYear);

        const monthsInYear = availableMonths.filter(month => month.startsWith(`${value}-`));
        if (monthsInYear.length > 0) {
          const lastMonth = monthsInYear[monthsInYear.length - 1];
          setSelectedMonth(parseInt(lastMonth.substring(5), 10));
        } else {
          setSelectedMonth(null);
        }
      });
    } else {
      if (!selectedYear) return;
      const options = monthsForSelectedYear.map(month => ({
        value: parseInt(month, 10).toString(),
        label: `${parseInt(month, 10)}월`,
      }));
      const currentValue = selectedMonth ? selectedMonth.toString() : '';
      floatingSelectRef.current?.open(options, currentValue, position, value => {
        setSelectedMonth(parseInt(value, 10));
      });
    }
  };

  const handleSelectClick = (event: React.MouseEvent, type: 'year' | 'month') => {
    openFloatingSelect(event.currentTarget as HTMLElement, type);
  };

  const openRangePopup = () => {
    if (!rangeSelectorRef.current) return;
    const rect = rangeSelectorRef.current.getBoundingClientRect();
    const mainContent = document.querySelector('.main-content') as HTMLElement | null;
    const mainWidth = mainContent ? mainContent.getBoundingClientRect().width : window.innerWidth;
    const viewportWidth = window.innerWidth;
    const maxAllowableWidth = Math.min(mainWidth * 0.7, viewportWidth - 24);
    const popupWidth = Math.max(
      rect.width,
      Math.min(sliderBaseWidth + 64, Math.max(320, maxAllowableWidth))
    );
    const rawLeft = rect.left + rect.width / 2 - popupWidth / 2;
    const left = Math.max(12, Math.min(rawLeft, viewportWidth - popupWidth - 12));
    const top = rect.bottom + 12;
    setRangePopupPos({ top, left, width: popupWidth });
    setIsRangePopupOpen(true);
  };

  const handleRangeSelectorClick = () => {
    if (!availableMonths.length) return;
    if (isRangePopupOpen) {
      setIsRangePopupOpen(false);
    } else {
      openRangePopup();
    }
  };

  const handleRangeSelectorKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleRangeSelectorClick();
    }
    if (event.key === 'Escape') {
      setIsRangePopupOpen(false);
    }
  };

  useEffect(() => {
    if (!isRangePopupOpen) return;

    const handleOutsideInteraction = (event: Event) => {
      if (!rangePopupRef.current || !rangeSelectorRef.current) return;
      const target = event.target;
      const isNodeTarget = target instanceof Node;
      const clickedInsidePopup = isNodeTarget ? rangePopupRef.current.contains(target) : false;
      const clickedSelector = isNodeTarget ? rangeSelectorRef.current.contains(target) : false;
      if (!clickedInsidePopup && !clickedSelector) {
        setIsRangePopupOpen(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsRangePopupOpen(false);
      }
    };

    window.addEventListener('mousedown', handleOutsideInteraction);
    window.addEventListener('scroll', handleOutsideInteraction, true);
    window.addEventListener('resize', handleOutsideInteraction);
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('mousedown', handleOutsideInteraction);
      window.removeEventListener('scroll', handleOutsideInteraction, true);
      window.removeEventListener('resize', handleOutsideInteraction);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isRangePopupOpen]);

  return (
    <div className="dashboard-page">
      <header className="main-header dashboard-header">
        <div className="header-title-group">
          <h1>Dashboard</h1>
          <div className="dashboard-header-controls">
            <div className="dashboard-date-selector">
              <div
                className="date-part year"
                onClick={e => handleSelectClick(e, 'year')}
                role="button"
                tabIndex={0}
                onKeyDown={event => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    openFloatingSelect(event.currentTarget as HTMLElement, 'year');
                  }
                }}
              >
                {selectedYear ?? '--'}
              </div>
              <span className="separator">년</span>
              <div
                className="date-part month"
                onClick={e => handleSelectClick(e, 'month')}
                role="button"
                tabIndex={0}
                onKeyDown={event => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    openFloatingSelect(event.currentTarget as HTMLElement, 'month');
                  }
                }}
              >
                {selectedMonth ?? '--'}
              </div>
              <span className="separator">월</span>
            </div>
            <div className="dashboard-header-divider" aria-hidden="true" />
            <div
              ref={rangeSelectorRef}
              className={`dashboard-range-display ${availableMonths.length ? '' : 'is-disabled'}`}
              role="button"
              tabIndex={availableMonths.length ? 0 : -1}
              aria-label="기간 선택"
              aria-expanded={isRangePopupOpen}
              aria-disabled={availableMonths.length === 0}
              onClick={handleRangeSelectorClick}
              onKeyDown={handleRangeSelectorKeyDown}
            >
              <span className="range-label">{rangeLabel}</span>
              <span className="range-caret" aria-hidden="true">▾</span>
            </div>
          </div>
        </div>
      </header>

      <FloatingSelectPopup ref={floatingSelectRef} />

      {isRangePopupOpen && (
        <div
          ref={rangePopupRef}
          className="dashboard-range-popup"
          style={{ top: rangePopupPos.top, left: rangePopupPos.left, width: rangePopupPos.width }}
        >
          <div className="popup-title">기간 선택</div>
          <div className="slider-wrapper">
            <div className="slider-container" ref={sliderContainerRef}>
              <div className="slider-track" style={{ width: sliderBaseWidth }}>
                <Slider
                  range
                  min={0}
                  max={availableMonths.length - 1}
                  value={range}
                  onChange={handleSliderChange}
                  onChangeComplete={handleSliderChangeComplete}
                  marks={sliderMarks}
                  step={1}
                  allowCross={false}
                // dotStyle과 activeDotStyle을 사용하여 점의 기본 스타일을 지정할 수 있습니다.
                // CSS에서 .rc-slider-dot으로 제어하므로 여기서는 생략합니다.
                />
              </div>
            </div>
          </div>
        </div>
      )}

      <main className="dashboard-content">
        <MonthlyDetail selectedYear={selectedYear} selectedMonth={selectedMonth} />
        <MonthlyTrend months={availableMonths} range={range} />
        <ComingSoon title="소비 습관 히트맵" />
        <ComingSoon title="소비 TOP 10" />
      </main>
    </div>
  );
};

export default Dashboard;