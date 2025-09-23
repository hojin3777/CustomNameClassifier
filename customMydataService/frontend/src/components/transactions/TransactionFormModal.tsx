import React, { useState, useEffect, useRef, useMemo } from 'react';
import DatePicker, { registerLocale } from 'react-datepicker';
import { ko } from 'date-fns/locale'; // 한국어 로케일
import { FaCalendarAlt } from 'react-icons/fa';
import type { Transaction, Appdata } from '../../pages/Transactions'; // Transactions.tsx의 타입 import
import type { FloatingSelectHandle, Opt } from './FloatingSelectPopup';
import FloatingSelectPopup from './FloatingSelectPopup';
import './DatePickerOverrides.css';
import './TransactionFormModal.css';
registerLocale('ko', ko);

interface TransactionFormModalProps {
  isOpen: boolean;
  onClose: () => void;
  onInsert: (newTransactions: Partial<Transaction>[]) => void;
  appData: Appdata;
  allTransactions: Transaction[];
}

const TransactionFormModal: React.FC<TransactionFormModalProps> = ({
  isOpen,
  onClose,
  onInsert,
  appData,
  allTransactions,
}) => {
  // ******************** 상태 관리 ********************
  // Refs
  const modalRef = useRef<HTMLDivElement>(null);
  const floatingSelectRef = useRef<FloatingSelectHandle>(null);
  const amountInputRef = useRef<HTMLInputElement>(null);

  // 드래그 상태
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // 데이터 상태
  const [date, setDate] = useState(new Date());
  const [selectedAccount, setSelectedAccount] = useState<number | null>(null);
  const [selectedType, setSelectedType] = useState<string>('유동지출');
  const [selectedMajor, setSelectedMajor] = useState<string | null>(null);
  const [selectedMinor, setSelectedMinor] = useState<string | null>(null);
  const [merchant, setMerchant] = useState('');
  const [amount, setAmount] = useState<number | null>(null);
  const [memo, setMemo] = useState('');
  const [toAccountId, setToAccountId] = useState<number | null>(null);

  // UI관련 상태
  const [merchantSuggestions, setMerchantSuggestions] = useState<Transaction[]>([]);
  const [isInternalTransfer, setIsInternalTransfer] = useState(false);
  const [amountSignWarning, setAmountSignWarning] = useState<string | null>(null);
  const prevSelectedType = useRef<string>(selectedType); // 이전 selectedType 저장용

  // ******************** 이벤트 핸들러 ********************
  // 드래그 핸들러
  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest('button, input, select, .selectable-list, .react-datepicker')) return; // 버튼, 입력창, 리스트, 달력 클릭 시 드래그 방지
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (isDragging) setPosition({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
  };

  const handleMouseUp = () => setIsDragging(false);

  useEffect(() => {
    if (isOpen && modalRef.current) {
      const { innerWidth, innerHeight } = window;
      const { offsetWidth, offsetHeight } = modalRef.current;
      setPosition({ x: (innerWidth - offsetWidth) / 2, y: (innerHeight - offsetHeight) / 3 });
    }
  }, [isOpen]);


  // ********************* 편집 관련 핸들러 ********************
  // 날짜 변경 핸들러
  const openDatePopup = (e: React.MouseEvent, part: 'year' | 'month' | 'day') => {
    const target = e.currentTarget as HTMLElement;
    const rect = target.getBoundingClientRect();
    let options: Opt[] = [];
    const currentYear = new Date().getFullYear();
    if (part === 'year') {
      options = [...Array(10)].map((_, i) => ({ value: String(currentYear - 5 + i), label: String(currentYear - 5 + i) }));
    } else if (part === 'month') {
      options = [...Array(12)].map((_, i) => ({ value: String(i + 1), label: String(i + 1) }));
    } else if (part === 'day') {
      const daysInMonth = new Date(date.getFullYear(), date.getMonth() + 1, 0).getDate();
      options = [...Array(daysInMonth)].map((_, i) => ({ value: String(i + 1), label: String(i + 1) }));
    }
    const onSelect = (v: string) => {
      const newDate = new Date(date);
      const numValue = Number(v);
      if (part === 'year') newDate.setFullYear(numValue);
      if (part === 'month') newDate.setMonth(numValue - 1); // JS의 월은 0-11
      if (part === 'day') newDate.setDate(numValue);
      setDate(newDate);
    };
    floatingSelectRef.current?.open(options, String(target.textContent), { top: rect.bottom, left: rect.left, width: rect.width }, onSelect);
  };

  // FloatingSelectPopup 열기 핸들러
  const openSelectPopup = (e: React.MouseEvent, options: Opt[], currentValue: string | number | null, onSelect: (value: string) => void) => {
    const target = e.currentTarget as HTMLElement;
    const rect = target.getBoundingClientRect();
    floatingSelectRef.current?.open(options, String(currentValue), { top: rect.bottom, left: rect.left, width: rect.width }, onSelect);
  };

  // 금액 입력 핸들러
  const handleAmountChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const rawValue = e.target.value;
    const sanitizedValue = rawValue.replace(/[^\d,.+-]/g, '').replace(/(?!^)-/g, '');
    if (amountInputRef.current) amountInputRef.current.value = sanitizedValue;
  };

  // 금액 입력 완료 핸들러
  const finalizeAmount = () => {
    const rawValue = amountInputRef.current?.value || '';
    const isPositive = rawValue.trim().startsWith('+');
    const numValue = parseFloat(rawValue.replace(/,/g, ''));
    if (isNaN(numValue) || numValue === 0) {
      setAmount(null);
      return;
    }
    if (!isPositive && numValue > 0){
      setAmount(-numValue);
    } else {
      setAmount(isPositive ? Math.abs(numValue) : numValue);
    }
  };

  // 금액 입력 필드에서 키보드 이벤트 처리
  const handleAmountKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      finalizeAmount();
      e.preventDefault(); // 폼 제출 방지
      // 다음 입력 필드(메모)로 포커스 이동
      const form = e.currentTarget.form;
      if (form) {
        const inputs = Array.from(form.elements) as HTMLElement[];
        const currentIndex = inputs.indexOf(e.currentTarget);
        const nextInput = inputs[currentIndex + 1];
        if (nextInput) {
          nextInput.focus();
        }
      }
    } else if (e.key === 'Escape') {
      setAmount(null); // Esc 누르면 입력 취소
      e.currentTarget.blur();
    }
  };

  // 금액 입력창 색상 클래스
  const amountColorClass = useMemo(() => {
    if (amount === null) return '';
    return amount >= 0 ? 'income' : 'expense';
  }, [amount]);

  // 연쇄선택 로직
  const availableMajors = useMemo(() => {
    const INCOME_TYPES = ['고정수입', '유동수입'];
    const EXPENSE_TYPES = ['고정지출', '반고정지출', '유동지출'];

    if (selectedType === '수입') return appData.categories.filter(c => INCOME_TYPES.includes(c.name));
    if (selectedType === '이체') return appData.categories.filter(c => c.name === '이체분류');
    if (INCOME_TYPES.includes(selectedType)) return appData.categories.filter(c => INCOME_TYPES.includes(c.name));
    if (EXPENSE_TYPES.includes(selectedType)) {
      const coreCategories = ['고정수입', '유동수입', '이체분류'];
      return appData.categories.filter(c => !coreCategories.includes(c.name));
    }
    return [];
  }, [selectedType, appData.categories]);

  const availableMinors = useMemo(() => {
    if (!selectedMajor) return [];
    const major = appData.categories.find(c => c.name === selectedMajor);
    return major ? major.minors : [];
  }, [selectedMajor, appData.categories]);

  // 기본값 설정 로직
  useEffect(() => {
    if (isOpen){
      setDate(new Date());
      if (appData.accounts.length > 0) setSelectedAccount(appData.accounts[0].id);
      setSelectedType('유동지출');
      setMerchant('');
      setAmount(null);
      setMemo('');
      setToAccountId(null);
      setAmountSignWarning(null);
    }
  }, [isOpen, appData.accounts]);

  useEffect(() => { //대분류 자동선택 로직 개선
    const EXPENSE_TYPES = ['고정지출', '반고정지출', '유동지출'];
    const prevIsExpense = EXPENSE_TYPES.includes(prevSelectedType.current);
    const currentIsExpense = EXPENSE_TYPES.includes(selectedType);

    // 이전에 선택된 대분류가 현재 사용 가능한 대분류 목록에 여전히 존재하는지 확인
    const isMajorStillAvailable = availableMajors.some(m => m.name === selectedMajor);

    // 지출 유형끼리 변경되고, 현재 대분류가 유효하다면 초기화하지 않음
    if (prevIsExpense && currentIsExpense && isMajorStillAvailable) {
      // 아무것도 하지 않음
    } else if (availableMajors.length > 0) {
      setSelectedMajor(availableMajors[0].name);
    } else {
      setSelectedMajor(null);
    }

    // 현재 타입을 이전 타입으로 업데이트
    prevSelectedType.current = selectedType;
  }, [selectedType, availableMajors]); // selectedType 변경 시에도 동작하도록 수정

  useEffect(() => {
    if (availableMinors.length > 0){
      setSelectedMinor(availableMinors[0].uuid);
    } else {
      setSelectedMinor(null);
    }
  }, [availableMinors]);

  useEffect(() => {
    const minor = availableMinors.find(m => m.uuid === selectedMinor);
    setIsInternalTransfer(minor?.name === '내계좌이체' || minor?.name === '저축');
    if (!isInternalTransfer) setToAccountId(null);
  }, [selectedMinor, availableMinors, isInternalTransfer]);

  // 금액 부호 경고 로직
  useEffect(() => {
    if (document.activeElement === amountInputRef.current){
      setAmountSignWarning(null);
      return;
    }
    if (amount === null) {
      setAmountSignWarning(null);
      return;
    }
    const isIncomeType = ['고정수입', '유동수입'].includes(selectedType);
    const isExpenseType = ['고정지출', '반고정지출', '유동지출'].includes(selectedType);
    if (isIncomeType && amount < 0) {
      setAmountSignWarning('수입 유형은 양수여야 합니다.');
    } else if (isExpenseType && amount > 0) {
      setAmountSignWarning('지출 유형은 음수여야 합니다.');
    } else {
      setAmountSignWarning(null);
    }
  }, [amount, selectedType]);



  // 거래처 검색 및 자동완성 로직
  // const handleMerchantKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
  //   if (e.key === 'Enter' && merchant) {
  //     const suggestions = allTransactions.filter(t => t.merchant.includes(merchant));
  //     const uniqueSuggestions = suggestions.filter((v, i, a) => a.findIndex(t => t.merchant === v.merchant) === i);
  //     setMerchantSuggestions(uniqueSuggestions);
  //   }
  // };

  // const handleSuggestionClick = (suggestion: Transaction) => {
  //   const majorCategory = appData.categories.find(c => c.minors.some(m => m.uuid === suggestion.minor_category_uuid));
  //   setMerchant(suggestion.merchant);
  //   setSelectedAccount(suggestion.account_id);
  //   setSelectedType(suggestion.type);
  //   if (majorCategory) setSelectedMajor(majorCategory.name);
  //   setSelectedMinor(suggestion.minor_category_uuid);
  //   setMerchantSuggestions([]);
  //   amountInputRef.current?.focus();
  // };

  // 입력 버튼 핸들러
  const handleInsert = () => {
    if (!selectedAccount || !selectedType || !selectedMajor || !selectedMinor || amount === null || amount === 0) {
      alert('필수 항목을 모두 입력해주세요.');
      return;
    }
    if (amountSignWarning) {
      alert(amountSignWarning);
      return;
    }
    const newTransaciton: Partial<Transaction> = {
      id: `tmp-${crypto.randomUUID()}`,
      transaction_date: date.toISOString().split('T')[0],
      account_id: selectedAccount,
      type: selectedType as Transaction['type'],
      major_category_name: selectedMajor,
      minor_category_uuid: selectedMinor,
      merchant: merchant,
      amount: amount,
      memo: memo,
      checked: false,
      is_bold: 0,
      flag_color_id: 0,
      highlight_color_id: 0,
      background_color_id: 0,
    };
    if (isInternalTransfer && toAccountId) {
      const transferTransaction: Partial<Transaction> = {
        ...newTransaciton,
        id: `tmp-${crypto.randomUUID()}`,
        account_id: toAccountId,
        amount: -amount,
        type: amount > 0 ? '이체' : '이체',
      };
      onInsert([newTransaciton, transferTransaction]);
    } else {
      onInsert([newTransaciton]);
    }
    onClose();
  };

  if (!isOpen) return null;

  const modalStyle = { transform: `translate(${position.x}px, ${position.y}px)` };

  return (
    <>
      <div className="form-modal-bg" onMouseMove={handleMouseMove} onMouseUp={handleMouseUp}>
        <div ref={modalRef} style={modalStyle} className="form-modal-content" onMouseDown={handleMouseDown}>
          <div className="form-modal-header">
            <div className="date-selector">
              <DatePicker selected={date} onChange={(d: Date | null) => d && setDate(d)} customInput={<button className="calendar-btn"><FaCalendarAlt /></button>} locale="ko" popperClassName="dp-popper" />
              <div className="date-part year" onClick={(e) => openDatePopup(e, 'year')}>{date.getFullYear()}</div><span className="separator">년</span>
              <div className="date-part month" onClick={(e) => openDatePopup(e, 'month')}>{date.getMonth() + 1}</div><span className="separator">월</span>
              <div className="date-part day" onClick={(e) => openDatePopup(e, 'day')}>{date.getDate()}</div><span className="separator">일</span>
              <span className="weekday">{date.toLocaleDateString('ko-KR', { weekday: 'short' })}</span>요일
            </div>
            <button onClick={onClose} className="close-btn">×</button>
          </div>
          <div className="form-modal-body">
            <div className="grid-panel">
              <div className="left-column">
                <div className="panel">
                  <label>계좌</label>
                  <ul className="selectable-list">
                    {appData.accounts.map(acc => (
                      <li key={acc.id} className={selectedAccount === acc.id ? 'selected' : ''} onClick={() => setSelectedAccount(acc.id)}>{acc.name}</li>
                    ))}
                  </ul>
                </div>
                <div className="panel">
                  <label>유형</label>
                  <ul className="selectable-list fixed-height">
                    {['수입', '고정지출', '반고정지출', '유동지출', '이체'].map(t => (
                      <li key={t} className={selectedType === t ? 'selected' : ''} onClick={() => setSelectedType(t)}>{t}</li>
                    ))}
                  </ul>
                </div>
              </div>
              <div className="panel">
                <label>대분류</label>
                <ul className="selectable-list full-height">
                  {availableMajors.map(m => (
                    <li key={m.id} className={selectedMajor === m.name ? 'selected' : ''} onClick={() => setSelectedMajor(m.name)}>{m.name}</li>
                  ))}
                </ul>
              </div>
              <div className="panel">
                <label>소분류</label>
                <ul className="selectable-list full-height">
                  {availableMinors.map(m => (
                    <li key={m.uuid} className={selectedMinor === m.uuid ? 'selected' : ''} onClick={() => setSelectedMinor(m.uuid)}>{m.name}</li>
                  ))}
                </ul>
              </div>
            </div>
            <div className="detail-fields">
              {isInternalTransfer && (
                <div className="form-group internal-transfer-panel">
                  <label>받는 계좌 (To)</label>
                  <ul className="selectable-list horizontal">
                    {appData.accounts.filter(acc => acc.id !== selectedAccount).map(a => (
                      <li key={a.id} className={toAccountId === a.id ? 'selected' : ''} onClick={() => setToAccountId(a.id)}>{a.name}</li>
                    ))}
                  </ul>
                </div>
              )}
              <div className="form-group">
                <label className='detail-label'>거래처</label>
                <input type="text" placeholder="거래처 입력..." value={merchant} onChange={e => setMerchant(e.target.value)} />
              </div>
              <div className="form-group-inline">
                <div className="form-group">
                  <div className='label-with-warning'>
                  <label className='detail-label'>금액</label>
                  {amountSignWarning && <span className="amount-warning">{amountSignWarning}</span>}
                  </div>
                  <input
                    ref={amountInputRef}
                    type="text"
                    className={`amount-input ${amountColorClass}`}
                    placeholder="₩ 0"
                    key={amount}
                    defaultValue={amount === null ? '' : amount.toLocaleString()}
                    onChange={handleAmountChange}
                    onBlur={finalizeAmount}
                    onKeyDown={handleAmountKeyDown}
                  />
                </div>
                <div className="form-group">
                  <label className='detail-label'>메모</label>
                  <input type="text" value={memo} onChange={e => setMemo(e.target.value)} />
                </div>
              </div>
            </div>
          </div>
          <div className="form-modal-footer">
            <button className="btn-primary" onClick={handleInsert}>입력</button>
            <button onClick={onClose}>종료</button>
          </div>
        </div>
      </div>
      <FloatingSelectPopup ref={floatingSelectRef} />
    </>
  );
};

export default TransactionFormModal;