import React, { useState, useEffect, useMemo } from 'react'; // useRef 추가
import { FaSave, FaUndo, FaTrash, FaBold, FaHighlighter, FaCaretDown, FaArrowUp, FaArrowDown, FaFilter } from 'react-icons/fa'; // FaCaretDown 추가
import DatePicker from 'react-datepicker';
import "react-datepicker/dist/react-datepicker.css";
import { useLocation, useNavigate } from 'react-router-dom';

import './Transactions.css';
import FilterPopup from './FilterPopup'; // ✨ 1. FilterPopup 컴포넌트 import
import './FilterPopup.css'; // ✨ FilterPopup CSS import
import ConfirmPopup from './ConfirmPopup'; // ✨ ConfirmPopup 컴포넌트 import
import './ConfirmPopup.css'; // ✨ ConfirmPopup CSS import

// ... 타입 정의는 기존과 동일 ...
export type Transaction = {
  id: number;
  checked: boolean;
  date: string;
  account: string;
  type: '수입' | '고정지출' | '반고정지출' | '유동지출' | '이체';
  majorCategory: string;
  minorCategory: string;
  amount: number;
  payee: string;
  memo: string;
};

const Transactions = () => {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [originalTransactions, setOriginalTransactions] = useState<Transaction[]>([]);
  const [editingCell, setEditingCell] = useState<{ rowId: number; column: keyof Transaction | null } | null>(null);
  const [filters, setFilters] = useState<{ [key: string]: any[] }>({});
  const [popupPosition, setPopupPosition] = useState({ top: 0, left: 0 });
  const [activeFilter, setActiveFilter] = useState<{ column: keyof Transaction, name: string } | null>(null);
  const [sortConfig, setSortConfig] = useState<{ key: keyof Transaction; direction: 'asc' | 'desc' } | null>(null);
  const [confirmPopup, setConfirmPopup] = useState({isOpen: false, title: '', message: '', onConfirm: () => {},
    onCancel: (() => {}) as (() => void) | undefined, type: 'info' as 'info' | 'destructive',});
  const [appData, setAppData] = useState<{
    accounts: string[];
    categories: { major: string; minors: string[] }[];
  }>({ accounts: [], categories: []});
  const TRANSACTION_TYPES: Transaction['type'][] = ['수입', '고정지출', '반고정지출', '유동지출', '이체'];
  const isDirty = useMemo(() => 
    JSON.stringify(transactions) !== JSON.stringify(originalTransactions),
    [transactions, originalTransactions]
  );
  const navigate = useNavigate();
  const location = useLocation();

  // --- 데이터 로딩 ---
  // ✨ 1. 데이터 로딩 로직을 별도 함수로 분리 (재사용을 위해)
  const fetchAllData = async () => {
    // 거래내역 로딩
    try {
      const transResponse = await fetch('http://localhost:5000/api/transactions');
      if (!transResponse.ok) throw new Error('거래내역 로딩 실패');
      const transData = await transResponse.json();
      setTransactions(transData);
      setOriginalTransactions(transData);
    } catch (error) {
      console.error("거래내역 로딩 중 오류:", error);
      setTransactions([]);
      setOriginalTransactions([]);
    }

    // 계좌 및 카테고리 데이터 로딩
    try {
      const appDataResponse = await fetch('http://localhost:5000/api/categories');
      if (!appDataResponse.ok) throw new Error('카테고리 데이터 로딩 실패');
      const appData: { major: string; minors: string[] }[] = await appDataResponse.json();
      
      const accountsData = appData.find(item => item.major === '계좌');
      const categoriesData = appData.filter(item => item.major !== '계좌');

      setAppData({
        accounts: accountsData ? accountsData.minors : [],
        categories: categoriesData,
      });
    } catch (error) {
      console.error("카테고리 데이터 로딩 중 오류:", error);
    }
  };

  // --- 데이터 로딩 및 미저장 변경 경고 ---
  // ✨ 2. 중복된 useEffect를 하나로 통합합니다.
  useEffect(() => {
    fetchAllData();
  }, []); // 이 useEffect는 컴포넌트가 처음 마운트될 때 한 번만 실행됩니다.

  // 미저장 변경 경고 useEffect는 기존과 동일
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (isDirty) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [isDirty]);

  const handleSave = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/transactions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(transactions),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '저장에 실패했습니다.');
      }
      
      const savedTransactions = await response.json();
      setTransactions(savedTransactions);
      setOriginalTransactions(savedTransactions);
      alert('성공적으로 저장되었습니다.');
    } catch (error) {
      console.error(error);
      alert(`저장 중 오류가 발생했습니다: ${error}`);
    }
  };
  // ✨ 5. 초기화 핸들러 수정 (커스텀 팝업 사용)
  const handleReset = () => {
    setConfirmPopup({
      isOpen: true,
      title: '거래내역 초기화',
      message: '모든 거래내역을 삭제하고 초기 상태로 되돌리시겠습니까? 이 작업은 되돌릴 수 없습니다.',
      onConfirm: async () => {
        try {
          // 1. 백엔드에 초기화 요청
          const response = await fetch('http://localhost:5000/api/transactions/reset', { method: 'POST' });
          if (!response.ok) throw new Error('초기화에 실패했습니다.');

          // 2. 초기화 성공 후, 데이터를 다시 불러옴
          //    (그러면 백엔드에서 자동으로 기본값을 생성해 줄 것임)
          await fetchAllData();
          
          alert('거래내역이 초기화되었습니다.');
        } catch (error) {
          console.error(error);
          alert('초기화 중 오류가 발생했습니다.');
        } finally {
          // 3. 팝업 닫기
          setConfirmPopup(prev => ({ ...prev, isOpen: false }));
        }
      },
      onCancel: () => setConfirmPopup(prev => ({ ...prev, isOpen: false })),
      type: 'destructive',
    });
  };
  const handleAddRow = () => {
    const newId = Math.max(...transactions.map(t => t.id), 0) + 1;
    const newRow: Transaction = {
      id: newId,
      checked: false,
      date: new Date().toISOString().split('T')[0],
      account: '',
      type: '유동지출',
      majorCategory: '',
      minorCategory: '',
      amount: 0,
      payee: '',
      memo: '',
    };
    
  const checkedIds = transactions.filter(t => t.checked).map(t => t.id);

    if (checkedIds.length === 0) {
      // 선택된 행이 없으면 맨 아래에 추가
      setTransactions(prev => [...prev, newRow]);
    } else {
      // 선택된 행이 있으면, 가장 마지막 선택된 행의 인덱스를 찾음
      // transactions 배열은 순서가 보장되므로, 뒤에서부터 찾으면 마지막 선택 행임
      let lastCheckedIndex = -1;
      for (let i = transactions.length - 1; i >= 0; i--) {
        if (transactions[i].checked) {
          lastCheckedIndex = i;
          break;
        }
      }
      
      // 해당 인덱스 바로 다음에 새 행을 삽입
      if (lastCheckedIndex !== -1) {
        setTransactions(prev => {
          const newTransactions = [...prev];
          newTransactions.splice(lastCheckedIndex + 1, 0, newRow);
          return newTransactions;
        });
      }
    }
  };
  // ✨ 3. handleDeleteSelected 함수를 커스텀 팝업을 사용하도록 수정
  const handleDeleteSelected = () => {
    const selectedCount = transactions.filter(t => t.checked).length;
    if (selectedCount === 0) {
      setConfirmPopup({
        isOpen: true,
        title: '경고',
        message: '삭제할 행을 선택하세요.',
        onConfirm: () => setConfirmPopup({ ...confirmPopup, isOpen: false }),
        onCancel: undefined,
        type: 'info', // 일반 정보 팝업
      });
      return;
    }

    setConfirmPopup({
      isOpen: true,
      title: '삭제 확인',
      message: `${selectedCount}개의 행을 삭제하시겠습니까?`,
      onConfirm: () => {
        setTransactions(prev => prev.filter(t => !t.checked));
        setConfirmPopup({ ...confirmPopup, isOpen: false });
      },
      onCancel: () => setConfirmPopup({ ...confirmPopup, isOpen: false }),
      type: 'destructive', // ✨ 2. 파괴적인 동작임을 명시
    });
  };
  // ✨ 2. 개별 행 체크박스 토글 핸들러 추가
  const handleToggleCheck = (rowId: number) => {
    setTransactions(prev => prev.map(t => t.id === rowId ? { ...t, checked: !t.checked } : t));
  };
  // ✨ 3. 전체 선택 체크박스 토글 핸들러 추가
  const handleToggleCheckAll = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isChecked = e.target.checked;
    // 현재 화면에 보이는 행들만 전체선택/해제 대상으로 함
    const visibleIds = new Set(processedTransactions.map(t => t.id));
    setTransactions(prev =>
      prev.map(t =>
        visibleIds.has(t.id) ? { ...t, checked: isChecked } : t
      )
    );
  };
  // ✨ 3. 필터 적용 핸들러
  const handleApplyFilter = (columnKey: string, selectedValues: any[]) => {
    setFilters(prev => ({ ...prev, [columnKey]: selectedValues }));
  };
  const handleClearColumnFilter = (columnKey: string) => {
    // 필터 상태에서 해당 컬럼 제거
    setFilters(prev => {
      const newFilters = { ...prev };
      delete newFilters[columnKey];
      return newFilters;
    });
    // 만약 현재 정렬 기준이 이 컬럼이라면 정렬도 해제
    if (sortConfig?.key === columnKey) {
      setSortConfig(null);
    }
  };
  const handleClearAllFilters = () => {
    setFilters({});
    setSortConfig(null);
  };
  const handleSort = (key: keyof Transaction, direction: 'asc' | 'desc') => {
    setSortConfig({ key, direction });
    setActiveFilter(null); // 정렬 후 팝업 닫기
  };
  const processedTransactions = React.useMemo(() => {
    let filtered = transactions.filter(transaction => {
      return Object.entries(filters).every(([key, selectedValues]) => {
        if (selectedValues.length === 0) return true;
        const transactionValue = transaction[key as keyof Transaction];
        return selectedValues.includes(transactionValue);
      });
    });

    if (sortConfig !== null) {
      // sort()는 원본 배열을 변경하므로, 복사본을 만들어 정렬합니다.
      filtered = [...filtered].sort((a, b) => {
        const aValue = a[sortConfig.key];
        const bValue = b[sortConfig.key];
        if (aValue < bValue) return sortConfig.direction === 'asc' ? -1 : 1;
        if (aValue > bValue) return sortConfig.direction === 'asc' ? 1 : -1;
        return 0;
      });
    }

    return filtered;
  }, [transactions, filters, sortConfig]);

  // ✨ 6. 헤더 클릭 핸들러 (위치 계산 추가)
  const handleHeaderClick = (e: React.MouseEvent, columnKey: keyof Transaction, title: string) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setPopupPosition({ top: rect.bottom + window.scrollY, left: rect.left + window.scrollX });
    setActiveFilter({ column: columnKey, name: title });
  };

  // ✨ 7. 헤더 렌더링 로직 (시각적 피드백 추가)
  const renderHeader = (columnKey: keyof Transaction, title: string) => {
    const isFiltered = filters[columnKey] && filters[columnKey].length > 0;
    const sortDirection = sortConfig?.key === columnKey ? sortConfig.direction : null;

    return (
      <th>
        <div
          className={`th-content ${isFiltered ? 'filtered' : ''}`}
          onClick={(e) => handleHeaderClick(e, columnKey, title)}
        >
          <span>{title}</span>
          {sortDirection === 'asc' && <FaArrowUp className="sort-icon asc" />}
          {sortDirection === 'desc' && <FaArrowDown className="sort-icon desc" />}
          {!sortDirection && <FaCaretDown className="sort-icon" />}
        </div>
      </th>
    );
  };
  // 행 데이터 업데이트
  const handleUpdateCell = (rowId: number, column: keyof Transaction, value: any) => {
    setTransactions(prev =>
      prev.map(row => {
        if (row.id === rowId) {
          // ... 금액 처리 로직은 기존과 동일 ...
          if (column === 'amount') {
            const numValue = parseFloat(value.toString().replace(/,/g, ''));
            const finalValue = value.toString().startsWith('+') ? Math.abs(numValue) : -Math.abs(numValue);
            return { ...row, [column]: finalValue };
          }
          if (column === 'type') {
            return { ...row, type: value, majorCategory: '', minorCategory: '' };
          }
          if (column === 'majorCategory') {
            return { ...row, majorCategory: value, minorCategory: '' };
          }
          return { ...row, [column]: value };
        }
        return row;
      })
    );
  };
  
  // ✨ 4. renderCell 함수를 대폭 수정하여 각 타입에 맞는 편집 UI를 렌더링
  const renderCell = (transaction: Transaction, column: keyof Transaction) => {
    const isEditing = editingCell?.rowId === transaction.id && editingCell?.column === column;
    const commonProps = { onBlur: () => setEditingCell(null), autoFocus: true };

    if (isEditing) {
      switch (column) {
        case 'date':
          return (
            <td className="editing">
              <DatePicker
                selected={new Date(transaction.date)}
                // ✨ 1. (date: Date)를 (date: Date | null)로 변경
                onChange={(date: Date | null) => {
                  // ✨ 2. date가 null이 아닐 때만 실행하도록 if문 추가
                  if (date) {
                    handleUpdateCell(transaction.id, 'date', date.toISOString().split('T')[0]);
                    setEditingCell(null); // 날짜 선택 후 바로 편집 모드 종료
                  }
                }}
                dateFormat="yyyy-MM-dd"
                onBlur={commonProps.onBlur}
                autoFocus={commonProps.autoFocus}
              />
            </td>
          );
        case 'account':
          return (
            <td className="editing">
              <select {...commonProps} defaultValue={transaction.account} onChange={(e) => handleUpdateCell(transaction.id, 'account', e.target.value)}>
                <option value="" disabled>-- 선택 --</option>
                {/* ✨ 5. MOCK_ACCOUNTS 대신 상태(appData) 사용 */}
                {appData.accounts.map(acc => <option key={acc} value={acc}>{acc}</option>)}
              </select>
            </td>
          );
        case 'type':
          return (
            <td className="editing">
              <select {...commonProps} defaultValue={transaction.type} onChange={(e) => handleUpdateCell(transaction.id, 'type', e.target.value)}>
                <option value="" disabled>-- 선택 --</option>
                {/* ✨ 6. appData 대신 상수로 변경된 TRANSACTION_TYPES 사용 */}
                {TRANSACTION_TYPES.map(type => <option key={type} value={type}>{type}</option>)}
              </select>
            </td>
          );
        case 'amount':
          return (
            <td className="editing">
              <input
                {...commonProps}
                type="text" // '+' 기호를 입력받기 위해 text 타입 사용
                defaultValue={transaction.amount > 0 ? `+${transaction.amount}` : transaction.amount}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleUpdateCell(transaction.id, 'amount', e.currentTarget.value);
                    setEditingCell(null);
                  }
                }}
              />
            </td>
          );
        // ✨ 3. 대분류/소분류 편집 UI 추가
        case 'majorCategory': { // case 블록을 중괄호로 감싸서 변수 스코프 분리
          // ✨ 하드코딩된 배열 대신, DB에서 불러온 appData를 직접 사용합니다.
          const CORE_CATEGORIES = ['고정수입', '유동수입', '이체분류'];
          
          let availableMajors: string[] = [];
          if (transaction.type === '수입') {
            // '수입' 유형일 때는 '고정수입', '유동수입'만 필터링
            availableMajors = appData.categories
              .map(c => c.major)
              .filter(major => major === '고정수입' || major === '유동수입');
          } else if (transaction.type === '이체') {
            // '이체' 유형일 때는 '이체분류'만 필터링
            availableMajors = appData.categories
              .map(c => c.major)
              .filter(major => major === '이체분류');
          } else if (['고정지출', '반고정지출', '유동지출'].includes(transaction.type)) {
            // 그 외 지출 유형일 때는 핵심 카테고리 3개를 제외한 나머지를 필터링
            availableMajors = appData.categories
              .map(c => c.major)
              .filter(major => !CORE_CATEGORIES.includes(major));
          }

          return (
            <td className="editing">
              <select
                {...commonProps}
                defaultValue={transaction.majorCategory}
                onChange={(e) => handleUpdateCell(transaction.id, 'majorCategory', e.target.value)}
                disabled={!transaction.type}
              >
                <option value="" disabled>-- 선택 --</option>
                {availableMajors.map(major => <option key={major} value={major}>{major}</option>)}
              </select>
            </td>
          );
        }
       case 'minorCategory':
          if (!transaction.majorCategory) return <td></td>;
          // MOCK_CATEGORIES 대신 상태(appData) 사용
          const minors = appData.categories.find(c => c.major === transaction.majorCategory)?.minors || [];
          return (
            <td className="editing">
              <select
                {...commonProps}
                defaultValue={transaction.minorCategory}
                onChange={(e) => {
                  handleUpdateCell(transaction.id, 'minorCategory', e.target.value);
                  setEditingCell(null); // 선택 후 바로 편집 모드 종료
                }}
                disabled={!transaction.majorCategory}
              >
                <option value="" disabled>-- 선택 --</option>
                {/* ✨ 'availableMinors'를 바로 위에서 정의한 'minors'로 수정 */}
                {minors.map(minor => <option key={minor} value={minor}>{minor}</option>)}
              </select>
            </td>
          );
        default: // 거래처, 메모 등
          return (
            <td className="editing">
              <input
                {...commonProps}
                type="text"
                defaultValue={transaction[column] as string}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleUpdateCell(transaction.id, column, e.currentTarget.value);
                    setEditingCell(null);
                  }
                }}
              />
            </td>
          );
      }
    }

    // 편집 모드가 아닐 때 셀 표시
    const cellValue = transaction[column];
    let displayValue: React.ReactNode = cellValue;
    let className = '';

    // ✨ 1. 플레이스홀더 로직 수정
    if ((column === 'minorCategory' || column === 'majorCategory' || column === 'account') && !cellValue) {
      displayValue = <span className="placeholder">-- 선택 --</span>;
    } else if (column === 'payee' && !cellValue) {
      displayValue = <span className="placeholder">-- 입력 --</span>;
    } else if (column === 'amount') {
      const amount = cellValue as number;
      displayValue = amount.toLocaleString();
      className = amount > 0 ? 'amount-income' : 'amount-expense';
    }

    return (
      <td className={className} onClick={() => setEditingCell({ rowId: transaction.id, column })}>
        {displayValue}
      </td>
    );
  };

  
  return (
    <>
      {/* ✨ 6. Categories.tsx와 동일한 구조의 헤더로 변경 */}
      <header className="main-header">
        <div className="header-title-group">
          <h1>Transactions</h1>
          <div className="header-actions">
            {/* ✨ 3. disabled={!isDirty} 속성을 제거합니다. */}
            <button className="icon-button-round" onClick={handleSave} title="저장"><FaSave /></button>
            <button className="icon-button-round" onClick={handleReset} title="초기화"><FaUndo /></button>
          </div>
        </div>
      </header>
      <div className="content-area transactions-page">
        {/* 상단 툴바 */}
        <div className="transactions-toolbar card">
          <button onClick={handleAddRow}>행 삽입</button>
          <button onClick={handleDeleteSelected}><FaTrash /> 행 삭제</button>
          <button onClick={handleClearAllFilters}><FaFilter /> 전체 필터 해제</button>
          <div className="divider"></div>
          <button><FaBold /></button>
          <button><FaHighlighter /></button>
          <div className="divider"></div>
          <button className="primary">내역입력 폼 열기</button>
          <button className="primary">딥러닝 자동입력</button>
        </div>

        {/* 거래내역 테이블 */}
        <div className="table-container">
          <table>
            <thead>
              <tr>
                {/* TODO: 각 헤더에 필터 버튼 추가 */}
                <th>
                <input
                  type="checkbox"
                  onChange={handleToggleCheckAll}
                  // 보이는 행이 모두 체크되었을 때만 '전체 선택' 체크박스 활성화
                  checked={processedTransactions.length > 0 && processedTransactions.every(t => t.checked)}
                />
              </th>
                {renderHeader('date', '날짜')}
                {renderHeader('account', '계좌')}
                {renderHeader('type', '유형')}
                {renderHeader('majorCategory', '대분류')}
                {renderHeader('minorCategory', '소분류')}
                {renderHeader('amount', '금액')}
                {renderHeader('payee', '거래처')}
                {renderHeader('memo', '메모')}
              </tr>
            </thead>
            <tbody>
              {processedTransactions.map((transaction) => (
                <tr key={transaction.id}>
                  <td>
                  <input
                    type="checkbox"
                    checked={transaction.checked}
                    onChange={() => handleToggleCheck(transaction.id)}
                  />
                </td>
                  {renderCell(transaction, 'date')}
                  {renderCell(transaction, 'account')}
                  {renderCell(transaction, 'type')}
                  {renderCell(transaction, 'majorCategory')}
                  {renderCell(transaction, 'minorCategory')}
                  {renderCell(transaction, 'amount')}
                  {renderCell(transaction, 'payee')}
                  {renderCell(transaction, 'memo')}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {/* ✨ 7. 필터 팝업 조건부 렌더링 */}
        {activeFilter && (
          <FilterPopup
            columnKey={activeFilter.column}
            columnName={activeFilter.name}
            allValues={transactions.map(t => t[activeFilter.column])}
            appliedFilters={filters[activeFilter.column] || []}
            onApply={handleApplyFilter}
            onClose={() => setActiveFilter(null)}
            // ✨ 10. 새 props 전달
            onSort={handleSort}
            onClearFilter={handleClearColumnFilter}
            position={popupPosition}
          />
        )}
        {/* ✨ 4. 확인/경고 팝업 조건부 렌더링 */}
        <ConfirmPopup
          isOpen={confirmPopup.isOpen}
          title={confirmPopup.title}
          message={confirmPopup.message}
          onConfirm={confirmPopup.onConfirm}
          onCancel={confirmPopup.onCancel}
          type={confirmPopup.type}
        />
      </div>
    </>
  );
};

export default Transactions;