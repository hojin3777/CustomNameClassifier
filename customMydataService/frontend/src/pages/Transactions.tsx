import React, { useState, useEffect, useMemo, useRef } from 'react'; // useRef 추가
import { FaSave, FaUndo, FaTrash,
  FaBold, FaHighlighter, FaFlag, FaFillDrip,
  FaCaretDown, FaArrowUp, FaArrowDown, FaFilter,
  FaPlus, FaArrowRight, FaAngleDoubleDown } from 'react-icons/fa'; // 각 아이콘 로드
import DatePicker, { registerLocale } from 'react-datepicker';
import { useDirty } from '../App';

import './Transactions.css';
import "react-datepicker/dist/react-datepicker.css";
import "../components/transactions/DatePickerOverrides.css"; // DatePicker 커스텀 CSS
import FilterPopup from '../components/transactions/FilterPopup.tsx'; // 1. FilterPopup 컴포넌트 import
// import '../components/FilterPopup.css'; // FilterPopup CSS import
import ConfirmPopup from '../components/ConfirmPopup'; // ConfirmPopup 컴포넌트 import
import '../components/ConfirmPopup.css'; // ConfirmPopup CSS import
import HighlightPopup from '../components/transactions/HighlightPopup'; // ighlightPopup 컴포넌트 import
import '../components/transactions/HighlightPopup.css'; // HighlightPopup CSS import
import FloatingSelectPopup, { type FloatingSelectHandle, type Opt } from '../components/FloatingSelectPopup.tsx';
import '../components/FloatingSelectPopup.css';
import OcrImageUploadModal from '../components/transactions/OcrImageUploadModal';
import '../components/transactions/OcrImageUploadModal.css';
import OcrPreviewTableModal, {type TransactionRow as OcrPreviewRow} from '../components/transactions/OcrPreviewTableModal';
import TransactionFormModal from '../components/transactions/TransactionFormModal';
import { ko } from 'date-fns/locale';
registerLocale('ko', ko);

// ******* 타입 정의 *******
type Account = { id: number; name: string; };
type MinorCategory = { uuid: string; name: string; };
export type CategoryItem = { id: number; name: string; minors: MinorCategory[]; };

// 페이지용 데이터 타입
export type Transaction = {
  id: number | string;
  checked: boolean;
  transaction_date: string;
  type: '수입' | '고정지출' | '반고정지출' | '유동지출' | '이체';
  amount: number | null;
  merchant: string;
  memo: string;
  is_bold: number; // 0 또는 1
  flag_color_id: number; // 0 또는 1
  highlight_color_id: number; // 0~6
  background_color_id: number; // 0~6

  account_id: number | null; // 계좌 ID
  minor_category_uuid: string | null; // 소분류 UUID

  account_name: string | null;
  major_category_name: string | null;
  minor_category_name: string | null;
};

export type Appdata = {
  accounts: Account[];
  categories: CategoryItem[];
  mappings: { [key: number]: string };
}

const API_BASE_URL = 'http://localhost:5000'; // 백엔드 API 기본 URL
const TRANSACTION_TYPES: Transaction['type'][] = ['수입', '고정지출', '반고정지출', '유동지출', '이체'];


const Transactions = () => {
  // state 정의
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [originalTransactions, setOriginalTransactions] = useState<Transaction[]>([]);
  const [checkedRows, setCheckedRows] = useState<Set<number | string>>(new Set());
  const [appData, setAppData] = useState<Appdata>({ accounts: [], categories: [], mappings: {} });

  const [editingCell, setEditingCell] = useState<{ rowId: number | string; column: keyof Transaction | null } | null>(null);
  const editingCellRef = useRef<any>(null);
  const floatingSelectRef = useRef<FloatingSelectHandle | null >(null);
  const [filters, setFilters] = useState<{ [key: string]: any[] }>({});
  const [popupPosition, setPopupPosition] = useState({ top: 0, left: 0 });
  const [activeFilter, setActiveFilter] = useState<{ column: keyof Transaction, name: string } | null>(null);
  const [sortConfig, setSortConfig] = useState<{ key: keyof Transaction; direction: 'asc' | 'desc' } | null>(null);
  const [confirmPopup, setConfirmPopup] = useState({isOpen: false, title: '', message: '', onConfirm: () => {},
    onCancel: (() => {}) as (() => void) | undefined, type: 'alert' as 'input' | 'confirm' | 'alert' | 'destructive'});
  const [status, setStatus] = useState('Loading...');
  const tableContainerRef = useRef<HTMLDivElement>(null); // 테이블 컨테이너 참조
  
  // highlight 팝업 관련 state
  const [isColorPopupOpen, setColorPopupOpen] = useState(false);
  const [colorPopupPosition, setColorPopupPosition] = useState({ top: 0, left: 0 });
  const [activeStyleType, setActiveStyleType] = useState<'flag' | 'highlight' | 'background' | null>(null);

  // OCR 이미지 업로드 모달 관련 state
  const [ocrModalOpen, setOcrModalOpen] = useState(false);
  const ocrButtonRef = useRef<HTMLButtonElement>(null);
  const [ocrPreviewRows, setOcrPreviewRows] = useState<any[]>([]);
  const [ocrPreviewOpen, setOcrPreviewOpen] = useState(false);
  const [ocrLoading, setOcrLoading] = useState(false);
  const [ocrLoadingText, setOcrLoadingText] = useState('딥러닝 추출중');

  // 내역입력 폼 모달 관련 state
  const [isFormModalOpen, setIsFormModalOpen] = useState(false);
  const [insertedCount, setInsertedCount] = useState(0);
  const [lastInsertedFromFormId, setLastInsertedFromFormId] = useState<number | string | null>(null);

  // dirty state
  const dirtyContext = useDirty();
  const isDirty = dirtyContext?.isDirty ?? false;
  const setIsDirty = dirtyContext?.setIsDirty ?? (() => {});


  // ESC키로 팝업 닫기
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setActiveFilter(null);
        setColorPopupOpen(false);
        if (editingCell) {
          setEditingCell(null);
        }
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [editingCell]);

  // 외부 클릭 감지하여 편집 완료처리
  useEffect(() => {
    if(!editingCell) return;
    const handleOutsideClick = (event: MouseEvent) => {
      if (editingCellRef.current && !editingCellRef.current.contains(event.target as Node)) {
        const nodeName = editingCellRef.current.nodeName;
        if (nodeName === 'INPUT' ) {
          const value = editingCellRef.current.value;
          handleUpdateCell(editingCell.rowId, editingCell.column!, value);
        }
        setEditingCell(null);
      }
    };
    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, [editingCell]);


  // ******* dirty 상태 감지 *******
  useEffect(() => {
    const isDifferent = JSON.stringify(transactions) !== JSON.stringify(originalTransactions);
    setIsDirty(isDifferent);
  }, [transactions, originalTransactions, setIsDirty]);

  // ******* 언마운트/새로고침 방지 *******
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (isDirty) {
        e.preventDefault();
        e.returnValue = '저장되지 않은 변경 사항이 있습니다. 정말 페이지를 떠나시겠습니까?';
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isDirty]);

  // ******* 폼 새 행 추가 이벤트 처리 *******
  useEffect(() => {
    if (lastInsertedFromFormId) {
      const rowElement = document.getElementById(`row-${lastInsertedFromFormId}`);
      if (rowElement) {
        rowElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        rowElement.classList.add('highlight-new');
        setTimeout(() => {
          rowElement.classList.remove('highlight-new');
        }, 3000);
      }
      setLastInsertedFromFormId(null);
    }
  }, [lastInsertedFromFormId]);


  // ******* 데이터 로딩 *******
  const fetchAllData = async () => {
    setStatus('Loading...');
    try {
      const [transRes, accRes, catRes, mapRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/transactions`),
        fetch(`${API_BASE_URL}/api/accounts`),
        fetch(`${API_BASE_URL}/api/categories`),
        fetch(`${API_BASE_URL}/api/mappings`),
      ]);
      if (!transRes.ok) throw new Error('거래내역 로딩 실패');
      if (!accRes.ok) throw new Error('계좌 데이터 로딩 실패');
      if (!catRes.ok) throw new Error('카테고리 데이터 로딩 실패');
      if (!mapRes.ok) throw new Error('매핑 데이터 로딩 실패');

      const transData: Transaction[] = await transRes.json();
      const accData: Account[] = await accRes.json();
      const catData: CategoryItem[] = await catRes.json();
      const mapData: { [key: number]: string } = await mapRes.json();

      const processedTrans = transData.map(t => ({ ...t, checked: false }));
      setTransactions(processedTrans);
      setOriginalTransactions(processedTrans);
      setAppData({ accounts: accData, categories: catData, mappings: mapData });
      setCheckedRows(new Set());
      setStatus('Loaded successfully');
      setTimeout(() => setStatus(''), 3000);
      setTimeout(() => {
        handleScrollToBottom();
      }, 200); // 데이터 로딩 후 약간의 지연을 두고 스크롤
    } catch (error) {
      console.error("Data loading failed:", error);
      setStatus("Data loading failed");
    }
  };

  useEffect(() => { fetchAllData(); }, []);


  // ******* 데이터 저장/초기화 *******
  // 빈 셀 체크 후 저장
  const handleSave = async () => {
    let emptyCellFound = null;
    for (const t of transactions) {
      if (t.account_id === null) {
       emptyCellFound = { rowId: t.id, column: 'account_name' as keyof Transaction };
        break;
      }
      if (t.minor_category_uuid === null) {
        emptyCellFound = { rowId: t.id, column: 'minor_category_name' as keyof Transaction };
        break;
      }
      if (t.merchant === '') {  
        emptyCellFound = { rowId: t.id, column: 'merchant' as keyof Transaction };
        break;
      }
      if (t.amount === null) {
        emptyCellFound = { rowId: t.id, column: 'amount' as keyof Transaction };
        break;
      }
    }
    if (emptyCellFound) {
      setConfirmPopup({
        isOpen: true,
        type: 'confirm',
        title: '미입력 항목 경고',
        message: '필수 입력 항목이 비어 있습니다.\n해당 행을 제외하고 저장을 진행하시겠습니까?',
        onConfirm: () => {
          setConfirmPopup(prev => ({ ...prev, isOpen: false }));
          proceedSave();
        },
        onCancel: () => {
          scrollToAndHighlightCell(emptyCellFound);
          setConfirmPopup(prev => ({ ...prev, isOpen: false }));
        },
      });
      return;
    }
    proceedSave();
  };
  
  // 실제 저장 로직
  const proceedSave = async () => {
    setStatus('Saving...');
    try{
      const payload = transactions.map(t => ({
        id: t.id,
        transaction_date: t.transaction_date,
        type: t.type,
        amount: t.amount,
        merchant: t.merchant,
        memo: t.memo,
        account_id: t.account_id,
        minor_category_uuid: t.minor_category_uuid,
        is_bold: t.is_bold,
        flag_color_id: t.flag_color_id,
        highlight_color_id: t.highlight_color_id,
        background_color_id: t.background_color_id
    }));
    const response = await fetch(`${API_BASE_URL}/api/transactions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error('Save failed');

    const savedData = await response.json();
    const processedData = savedData.map((t: Transaction) => ({ ...t, checked: false }));
    setTransactions(processedData);
    setOriginalTransactions(processedData);
    setStatus('Saved successfully');
    setTimeout(() => setStatus(''), 3000);
  } catch (error) {
    console.error("Save failed:", error);
    setStatus("Save failed");
    await fetchAllData(); // 저장 실패 시 데이터 다시 불러오기
  }};

  // 빈 셀 스크롤
  const scrollToAndHighlightCell = (cellinfo: { rowId: number | string; column: keyof Transaction }) => {
    const cellId = `cell-${cellinfo.rowId}-${cellinfo.column}`;
    const cellElement = document.getElementById(cellId);
    const containerElement = tableContainerRef.current;
    if (cellElement && containerElement) {
      // element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      const duration = 200; // 스크롤 지속 시간 (ms). 이 값을 줄이면 더 빨라집니다.
      const start = containerElement.scrollTop;
      const end = cellElement.offsetTop - (containerElement.clientHeight / 2) + (cellElement.clientHeight / 2);
      const distance = end - start;
      let startTime: number | null = null;
      // 애니메이션을 실행하는 함수
      const step = (timestamp: number) => {
        if (!startTime) {
          startTime = timestamp;
        }
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1); // 0과 1 사이의 진행률
        const easeInOutQuad = progress < 0.5 ? 2 * progress * progress : -1 + (4 - 2 * progress) * progress; // 부드러운 시작과 끝을 위한 Easing 함수 적용
        containerElement.scrollTop = start + distance * easeInOutQuad;
        if (elapsed < duration) {
          requestAnimationFrame(step); // 애니메이션이 끝나지 않았으면 다음 프레임 요청
        }
      };
      requestAnimationFrame(step); // 애니메이션 시작
      cellElement.classList.add('highlight-error');
      setTimeout(() => {
        cellElement.classList.remove('highlight-error');
      }, 3000);
    }
  };


  // 초기화(모든 데이터 삭제) 핸들러
  const handleReset = () => {
    setConfirmPopup({
      isOpen: true,
      type: 'destructive',
      title: '거래내역 초기화',
      message: '모든 거래내역을 삭제하고 초기 상태로 되돌리시겠습니까?\n이 작업은 되돌릴 수 없습니다.',
      onConfirm: async () => {
        setConfirmPopup(prev => ({ ...prev, isOpen: false }));
        setStatus('Resetting...');
        try {
          const response = await fetch(`${API_BASE_URL}/api/transactions/reset`, { method: 'POST' });
          if (!response.ok) throw new Error('Reset failed.');
          await fetchAllData();
          setStatus('Reset successfully');
        } catch (error) {
          console.error("Reset failed:", error);
          setStatus("Reset failed");
        } finally {
          setTimeout(() => setStatus(''), 3000);
        }
      },
      onCancel: () => setConfirmPopup(prev => ({ ...prev, isOpen: false })),
    });
  };

  // ******* 툴박스 관련 핸들러 *******
  // 행 추가 핸들러
  const handleAddRow = () => {
    const newRow: Transaction = {
      id: `tmp-${crypto.randomUUID()}`, // 임시 ID
      checked: false,
      transaction_date: new Date().toISOString().split('T')[0],
      type: '유동지출',
      amount: null,
      merchant: '',
      memo: '',
      account_id: null,
      account_name: null,
      minor_category_uuid: null,
      major_category_name: null,
      minor_category_name: null,
      is_bold: 0,
      flag_color_id: 0,
      highlight_color_id: 0,
      background_color_id: 0
    };
    let lastCheckedIndex = -1;
    if (checkedRows.size > 0){
      for (let i = transactions.length - 1; i >= 0; i--) {
      if (checkedRows.has(transactions[i].id)) {
        lastCheckedIndex = i;
        break;
      }}
    }
    if (lastCheckedIndex === -1) { // 선택된 행이 없으면 맨 뒤에 추가
      setTransactions(prev => [...prev, newRow]);
    } else {
      setTransactions(prev => { // 선택된 행 뒤에 추가
        const newTransactions = [...prev];
        newTransactions.splice(lastCheckedIndex + 1, 0, newRow);
        return newTransactions;
      });
    }
  };

  // handleDeleteSelected 함수를 커스텀 팝업을 사용하도록 수정
  const handleDeleteSelected = () => {
    if (checkedRows.size === 0) {
      setConfirmPopup({
        isOpen: true,
        type: 'alert', // 일반 정보 팝업
        title: '알림',
        message: '삭제할 행을 선택하세요.',
        onConfirm: () => setConfirmPopup({ ...confirmPopup, isOpen: false }),
        onCancel: undefined,        
      });
      return;
    }
    setConfirmPopup({
      isOpen: true,
      type: 'confirm',
      title: '',
      message: `${checkedRows.size}개의 행을 삭제하시겠습니까?`,
      onConfirm: () => {
        setTransactions(prev => prev.filter(t => !checkedRows.has(t.id)));
        setCheckedRows(new Set());
        setConfirmPopup({ ...confirmPopup, isOpen: false });
      },
      onCancel: () => setConfirmPopup({ ...confirmPopup, isOpen: false }),      
    });
  };

  // 서식 지정 핸들러
  const handleApplyBold = () => {
    setTransactions(prev =>
      prev.map(t =>
        checkedRows.has(t.id) ? { ...t, is_bold: t.is_bold === 1 ? 0 : 1 } : t
      )
    );
  };
  const handleApplyColor = (colorId: number) => {
    if (!activeStyleType) return;
    const styleKey = `${activeStyleType}_color_id` as const;
    setTransactions(prev =>
      prev.map(t =>
        checkedRows.has(t.id) ? { ...t, [styleKey]: colorId } : t
      )
    );
    setColorPopupOpen(false); // 색상 선택 후 팝업 닫기
  };

  // 스타일 팝업 열기 핸들러
  const handleOpenColorPopup = (e: React.MouseEvent, type: 'flag' | 'highlight' | 'background') => {
    const rect = e.currentTarget.getBoundingClientRect();
    setColorPopupPosition({ top: rect.bottom + 5, left: rect.left });
    setActiveStyleType(type);
    setColorPopupOpen(true);
  };

  // 맨 아래로 스크롤 핸들러
  const handleScrollToBottom = () => {
    const element = tableContainerRef.current;
    if (!element) return;

    const duration = 200; // 스크롤 지속 시간 (ms). 이 값을 줄이면 더 빨라집니다.
    const start = element.scrollTop;
    const end = element.scrollHeight - element.clientHeight;
    const distance = end - start;
    let startTime: number | null = null;

    // 애니메이션을 실행하는 함수
    const step = (timestamp: number) => {
      if (!startTime) {
        startTime = timestamp;
      }
      const elapsed = timestamp - startTime;
      const progress = Math.min(elapsed / duration, 1); // 0과 1 사이의 진행률

      // 부드러운 시작과 끝을 위한 Easing 함수 적용
      const easeInOutQuad = progress < 0.5 ? 2 * progress * progress : -1 + (4 - 2 * progress) * progress;
      element.scrollTop = start + distance * easeInOutQuad;
      if (elapsed < duration) {
        requestAnimationFrame(step); // 애니메이션이 끝나지 않았으면 다음 프레임 요청
      }
    };
    requestAnimationFrame(step); // 애니메이션 시작
  };

  // 폼 입력 받는 핸들러
  const handleInsertTransactions = (newTransactions: Partial<Transaction>[]) => {
    const completeNewTransactions = newTransactions.map(t => {
      const account = appData.accounts.find(acc => acc.id === t.account_id);
      const majorCategory = appData.categories.find(cat => cat.name === t.major_category_name);
      const minorCategory = majorCategory?.minors.find(min => min.uuid === t.minor_category_uuid);

      return {
        ...t,
        id: t.id ?? `tmp-${crypto.randomUUID()}`,
        transaction_date: t.transaction_date || new Date().toDateString().split('T')[0],
        account_name: account?.name || null,
        account_id: t.account_id || 0,
        type: t.type || '유동지출',
        major_category_name: t.major_category_name || '',
        minor_category_uuid: t.minor_category_uuid || null,
        minor_category_name: minorCategory?.name || null,
        merchant: t.merchant || '',
        amount: t.amount || 0,
        memo: t.memo || '',
        checked: t.checked || false,
        is_bold: t.is_bold || 0,
        flag_color_id: t.flag_color_id || 0,
        highlight_color_id: t.highlight_color_id || 0,
        background_color_id: t.background_color_id || 0,
      };
    }) as Transaction[];
    const lastNewId = completeNewTransactions[completeNewTransactions.length - 1].id;
    setTransactions(prev => [...prev, ...completeNewTransactions]);
    setLastInsertedFromFormId(lastNewId);
  };

  const handleCloseFormModal = (finalInsertedCount: number) => {
    setIsFormModalOpen(false);
    if (finalInsertedCount > 0) {
      setConfirmPopup({
        isOpen: true,
        type: 'alert', // 일반 정보 팝업
        title: '',
        message: `${finalInsertedCount}개의 거래내역이 추가되었습니다.`,
        onConfirm: () => setConfirmPopup(prev => ({ ...prev, isOpen: false })),
        onCancel: undefined,
      });
    }
    setInsertedCount(0);
  };

  // ******* 셀 업데이트 로직 *******
  const handleUpdateCell = (rowId: number | string, column: keyof Transaction, value: any) => {
    setTransactions(prev =>
      prev.map(row => {
        if (row.id !== rowId) return row; // 변경 대상이 아니면 그대로 반환
        let newRow = { ...row, [column]: value }; // 변경 대상 행 복사
        if (column === 'account_id') {
          const selectedAccount = appData.accounts.find(acc => acc.id === value);
          newRow.account_name = selectedAccount ? selectedAccount.name : null;
        }
        if (column === 'type') {
          newRow.major_category_name = null;
          newRow.minor_category_uuid = null;
          newRow.minor_category_name = null;
        }
        if (column === 'major_category_name') {
          newRow.minor_category_uuid = null;
          newRow.minor_category_name = null;
        }
        if (column === 'minor_category_uuid') {
          for (const major of appData.categories) {
            const selectedMajor = major.minors.find(min => min.uuid === value);
            if (selectedMajor) {
              newRow.minor_category_name = selectedMajor.name;
              break;
            }
          }
        }
        if (column === 'amount') {
          const strValue = String(value);
          const isIncomeAmount = strValue.startsWith('+');
          const numValue = parseInt(strValue.replace(/[+,]/g, ''), 10) || 0;
          newRow.amount = isIncomeAmount ? numValue : -Math.abs(numValue);
          // 금액과 유형이 맞지 않는경우 알림 출력
          const type = newRow.type;
          const expenseTypes = ['고정지출', '반고정지출', '유동지출'];
          const condition1 = type === '수입' && !isIncomeAmount; // 수입 금액인데 유형이 수입이 아님(이체는 제외)
          const condition2 = expenseTypes.includes(type) && isIncomeAmount; // 지출 금액인데 유형이 지출이 아님(이체는 제외)
          
          if (condition1 || condition2) {
            setConfirmPopup({
              isOpen: true,
              type: 'alert', // 일반 정보 팝업
              title: '유형/금액 불일치 경고',
              message: `유형(${type})과 금액(${isIncomeAmount ? '양수 값' : '음수 값'})이 일치하지 않습니다.`,
              onConfirm: () => setConfirmPopup({ ...confirmPopup, isOpen: false }),
              onCancel: undefined,
            });
          }
        }
        return newRow;
      })
    );
  };
  // 달력 helper
  const parseYMD = (ymd?: string | null) => {
    if (!ymd) return null;
    const parts = ymd.split('-').map(part => parseInt(part, 10));
    if (parts.length !== 3 || parts.some(isNaN)) return null;
    return new Date(parts[0], parts[1] - 1, parts[2]);
  }


  // ******* 체크박스 관련 핸들러 *******
  // 개별 행 체크박스 토글 핸들러
  const handleToggleCheck = (rowId: number | string) => {
    setCheckedRows(prev => {
      const newSet = new Set(prev);
      if (newSet.has(rowId)) {
        newSet.delete(rowId);
      } else {
        newSet.add(rowId);
      }
      return newSet;
    });
  };
  // 선택된 행이 하나라도 있는지 여부
  const hasCheckedRows = checkedRows.size > 0;
  // 전체 선택 체크박스 토글 핸들러
  const handleToggleCheckAll = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isChecked = e.target.checked;
    if (isChecked) {
      // 현재 화면에 보이는 행들만 전체선택/해제 대상으로 함
      const visibleIds = new Set(processedTransactions.map(t => t.id));
      setCheckedRows(visibleIds);
    } else {
      setCheckedRows(new Set());
    };
  };


  // ******* 필터/정렬 관련 핸들러 *******
  // 3. 필터 적용 핸들러
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
        if ( aValue === null || aValue === undefined ) return 1;
        if ( bValue === null || bValue === undefined ) return -1;
        if (aValue < bValue) return sortConfig.direction === 'asc' ? -1 : 1;
        if (aValue > bValue) return sortConfig.direction === 'asc' ? 1 : -1;
        return 0;
      });
    }
    return filtered;
  }, [transactions, filters, sortConfig]);

  // 헤더 클릭 핸들러 (위치 계산 추가)
  const handleHeaderClick = (e: React.MouseEvent, columnKey: keyof Transaction, title: string) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setPopupPosition({ top: rect.bottom + window.scrollY, left: rect.left + window.scrollX });
    setActiveFilter({ column: columnKey, name: title });
  };


  // ******* 딥러닝 자동입력 관련 핸들러 *******
  const handleOcrUpload = async (files: File[]) => {
    setOcrModalOpen(false);
    setOcrLoading(true);
    try {
      const formData = new FormData();
      files.forEach(f => formData.append('images', f)); // 반드시 'images'!
      const res = await fetch(`${API_BASE_URL}/api/ocr/transactions`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('API 요청 실패');
      const data = await res.json();
      setOcrPreviewRows(data);
      setOcrPreviewOpen(true);
    } catch (e) {
      alert('OCR 추출에 실패했습니다: ' + (e as Error).message);
      console.error(e);
    } finally {
      setOcrLoading(false);
    }
  };
  // 1. 로딩 애니메이션을 위한 useEffect 추가
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (ocrLoading) {
      let dotCount = 0;
      interval = setInterval(() => {
        dotCount = (dotCount + 1) % 4; // 0, 1, 2, 3
        const dots = '.'.repeat(dotCount);
        setOcrLoadingText(`딥러닝 추출중${dots}`);
      }, 400);
    }
    return () => clearInterval(interval); // 컴포넌트 언마운트 또는 ocrLoading이 false가 되면 인터벌 정리
  }, [ocrLoading]);

  // OCR 미리보기 삽입 핸들러
  const handleOcrInsert = (rowsToInsert: OcrPreviewRow[]) => {
    const newTransactions = rowsToInsert.map(row => ({
      ...row,
      id: `tmp-${crypto.randomUUID()}`, // 메인 테이블용 새 임시 ID
      checked: false,
      is_bold: 0,
      flag_color_id: 0,
      highlight_color_id: 0,
      background_color_id: 0,
      type: row.type as Transaction['type'],
    }));
    setTransactions(prev => [...prev, ...newTransactions]);
    setOcrPreviewOpen(false);
  };




  // ******* 렌더링 헬퍼 함수 *******
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
  
  const renderCell = (transaction: Transaction, column: keyof Transaction) => {
    const isEditing = editingCell?.rowId === transaction.id && editingCell?.column === column;
    // const handleBlur = (e: React.FocusEvent<HTMLInputElement | HTMLSelectElement>) => {
    //   if (['amount', 'merchant', 'memo'].includes(column)) {
    //     handleUpdateCell(transaction.id, column, (e.target as HTMLInputElement).value);
    //   }
    //   setEditingCell(null);
    // };
    // const commonProps = { onBlur: () => handleBlur, autoFocus: true };
    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        handleUpdateCell(transaction.id, column, e.currentTarget.value);
        setEditingCell(null);
      } else if (e.key === 'Escape') {
        setEditingCell(null);
      }
    };

    if (isEditing) {
      const commonProps = { ref:editingCellRef, autoFocus: true }
      switch (column) {
        case 'transaction_date':
          return (
            <td className="editing">
              <DatePicker
                selected={new Date(transaction.transaction_date)}
                onChange={(date: Date | null) => {
                  if (date) {
                    const y = date.getFullYear();
                    const m = String(date.getMonth() + 1).padStart(2, '0');
                    const d = String(date.getDate()).padStart(2, '0');
                    handleUpdateCell(transaction.id, 'transaction_date', `${y}-${m}-${d}`);
                    setEditingCell(null); // 날짜 선택 후 바로 편집 모드 종료
                  }
                }}
                dateFormat="yyyy-MM-dd"
                onCalendarClose={() => setEditingCell(null)}
                onClickOutside={() => setEditingCell(null)}
                onBlur={() => setEditingCell(null)}
                autoFocus={commonProps.autoFocus}
                className='dp-input'
                locale={ko}
                popperClassName='dp-popper'
                calendarClassName='dp-calendar'
                dayClassName={(date) => {
                  const d = date;
                  const formatted = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
                  return formatted === transaction.transaction_date ? 'dp-day-selected' : '';
                }}
              />
            </td>
          );
        case 'account_name':
          return (
            <td className="editing">
              <input {...commonProps} type="text" defaultValue={transaction.account_name ?? ''} onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === 'Escape') setEditingCell(null); 
              }} />
              {/* <select {...commonProps} value={transaction.account_id ?? ''} onChange={(e) => {
                handleUpdateCell(transaction.id, 'account_id', parseInt(e.target.value));
                setEditingCell(null);
              }}>
                <option value="" disabled>-- 계좌 선택 --</option>
                {appData.accounts.map(acc => <option key={acc.id} value={acc.id}>{acc.name}</option>)}
              </select> */}
            </td>
          );
        case 'type':
          return (
            <td className="editing">
              <input {...commonProps} type="text" defaultValue={transaction.type} onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === 'Escape') setEditingCell(null);
              }} />
              {/* <select {...commonProps} value={transaction.type} onChange={(e) => {
                handleUpdateCell(transaction.id, 'type', e.target.value);
                setEditingCell(null);
              }}>
                <option value="" disabled>-- 선택 --</option>
                {TRANSACTION_TYPES.map(type => <option key={type} value={type}>{type}</option>)}
              </select> */}
            </td>
          );
        case 'major_category_name': {
          const INCOME_CATEGORIES = ['고정수입', '유동수입'];
          const TRANSFER_CATEGORY = '이체분류';
          const CORE_CATEGORIES = [...INCOME_CATEGORIES, TRANSFER_CATEGORY];
          let availableMajors: CategoryItem[] = [];
          if (transaction.type === '수입') { // '수입' 유형일 때는 '고정수입', '유동수입'만 필터링
            availableMajors = appData.categories.filter(c => INCOME_CATEGORIES.includes(c.name));
          } else if (transaction.type === '이체') {
            availableMajors = appData.categories.filter(c => c.name === TRANSFER_CATEGORY);
          } else if (['고정지출', '반고정지출', '유동지출'].includes(transaction.type)) {
            availableMajors = appData.categories.filter(c => !CORE_CATEGORIES.includes(c.name));
          }
          return (
            <td className="editing">
              <input {...commonProps} type="text" defaultValue={transaction.major_category_name ?? ''} onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === 'Escape') setEditingCell(null);
              }} />
              {/* <select
                {...commonProps}
                value={transaction.major_category_name ?? ''}
                onChange={(e) => {
                  handleUpdateCell(transaction.id, 'major_category_name', e.target.value);
                  setEditingCell(null);
                }}
                disabled={!transaction.type}
              >
                <option value="" disabled>-- 대분류 --</option>
                {availableMajors.map(major => <option key={major.id} value={major.name}>{major.name}</option>)}
              </select> */}
            </td>
          );
        }
        case 'minor_category_name': {
          const major = appData.categories.find(c => c.name === transaction.major_category_name);
          const availableMinors = major ? major.minors : [];
          return (
            <td className="editing">
              <input {...commonProps} type="text" defaultValue={transaction.minor_category_name ?? ''} onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === 'Escape') setEditingCell(null);
              }} />
              {/* <select
                {...commonProps}
                value={transaction.minor_category_uuid ?? ''}
                onChange={(e) => {
                  handleUpdateCell(transaction.id, 'minor_category_uuid', e.target.value);
                  setEditingCell(null); // 선택 후 바로 편집 모드 종료
                }}
                disabled={!transaction.major_category_name}
              >
                <option value="" disabled>-- 소분류 --</option>
                {availableMinors.map(minor => <option key={minor.uuid} value={minor.uuid}>{minor.name}</option>)}
              </select> */}
            </td>
          );
        }
        case 'amount':
          return (
            <td className="editing">
              <input
                {...commonProps}
                type="text" // '+' 기호를 입력받기 위해 text 타입 사용
                defaultValue={transaction.amount === null ? '' : (transaction.amount > 0 ? `+${transaction.amount}` : transaction.amount)}
                onKeyDown={handleKeyDown}
              />
            </td>
          );
        default: // 거래처, 메모 등
          return (
            <td className="editing">
              <input
                {...commonProps}
                type="text"
                defaultValue={transaction[column] as string}
                onKeyDown={handleKeyDown}
              />
            </td>
          );
      }
    }

    // 편집 모드가 아닐 때 셀 표시
    const cellValue = transaction[column];
    let displayValue: React.ReactNode = cellValue;
    let className = '';
    const cellId = `cell-${transaction.id}-${column}`;

    if (!cellValue && ['account_name', 'major_category_name', 'minor_category_name'].includes(column)) {
      const placeholderText = column === 'account_name' ? '-- 계좌 --' :
        column === 'major_category_name' ? '-- 대분류 --' : '-- 소분류 --';
      displayValue = <span className="placeholder">{placeholderText}</span>;
    } else if (column === 'amount') {
      const amount = cellValue as number | null;
      if (amount === null) {
        displayValue = <span className="placeholder">-- 금액 --</span>;
      } else {
        displayValue = amount.toLocaleString();
      className = amount >= 0 ? 'amount-income' : 'amount-expense';
      }
    } else if (column === 'merchant') {
        if (cellValue === '') {
          displayValue = <span className="placeholder">-- 거래처 --</span>;
        } else{0.
          displayValue = cellValue;
        }
        const merchantText = displayValue;
        return(
          <td id={cellId} onClick={() => setEditingCell({ rowId: transaction.id, column })}>
          {transaction.highlight_color_id > 0 ? (
            <span className={`text-highlight-${transaction.highlight_color_id}`}>
              {merchantText}
            </span>
          ):(merchantText)}
          </td>
        );
    }
    const onCellClick = (e: React.MouseEvent) => {
      if(['account_name', 'type', 'major_category_name', 'minor_category_name'].includes(String(column))) {
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        const pos = { top: rect.bottom + window.scrollY + 2, left: rect.left + window.scrollX, width: rect.width };
        // 옵션 생성 및 select open 콜백
        if (column === 'account_name') {
          const opts: Opt[] = appData.accounts.map(a => ({ value: String(a.id), label: a.name }));
          floatingSelectRef.current?.open(opts, String(transaction.account_id ?? ''), pos, (v: string) => {
            handleUpdateCell(transaction.id, 'account_id', v === '' ? null : Number(v));
          }, '-- 계좌 --');
          return;
        }
        if (column === 'type') {
          const opts: Opt[] = TRANSACTION_TYPES.map(t => ({ value: t, label: t }));
          floatingSelectRef.current?.open(opts, transaction.type, pos, (v: string) => {
            handleUpdateCell(transaction.id, 'type', v);
            // major/minor 초기화 handled in handleUpdateCell when type changes
          }, '-- 선택 --');
          return;
        }
        if (column === 'major_category_name') {
          const INCOME_CATEGORIES = ['고정수입', '유동수입'];
          const TRANSFER_CATEGORY = '이체분류';
          const CORE_CATEGORIES = [...INCOME_CATEGORIES, TRANSFER_CATEGORY];
          let availableMajors: CategoryItem[] = [];
          if (transaction.type === '수입') {
            availableMajors = appData.categories.filter(c => INCOME_CATEGORIES.includes(c.name));
          } else if (transaction.type === '이체') {
            availableMajors = appData.categories.filter(c => c.name === TRANSFER_CATEGORY);
          } else if (['고정지출', '반고정지출', '유동지출'].includes(transaction.type)) {
            availableMajors = appData.categories.filter(c => !CORE_CATEGORIES.includes(c.name));
          }
          const opts: Opt[] = availableMajors.map(m => ({ value: m.name, label: m.name }));
          floatingSelectRef.current?.open(opts, transaction.major_category_name ?? '', pos, (v: string) => {
            handleUpdateCell(transaction.id, 'major_category_name', v);
          }, '-- 대분류 --');
          return;
        }
        if (column === 'minor_category_name') {
          const major = appData.categories.find(c => c.name === transaction.major_category_name);
          const availableMinors = major ? major.minors : [];
          const opts: Opt[] = availableMinors.map(m => ({ value: m.uuid, label: m.name }));
          floatingSelectRef.current?.open(opts, transaction.minor_category_uuid ?? '', pos, (v: string) => {
            handleUpdateCell(transaction.id, 'minor_category_uuid', v === '' ? null : v);
          }, '-- 소분류 --');
          return;
        }
      }
      // 기본 텍스트 편집 동작
      setEditingCell({ rowId: transaction.id, column });
    };

    return (
      // <td id={cellId} className={className} onClick={() => setEditingCell({ rowId: transaction.id, column })}>
      <td id={cellId} className={className} onClick={onCellClick}>
        {displayValue}
      </td>
    );
  };

  
  // ******* 메인 렌더링 *******
  return (
    <>
    <div className='transactions-page'>
      <header className="main-header">
        <div className="header-title-group">
          <h1>Transactions</h1>
          <div className="header-actions">
            <button className={`icon-button-round ${isDirty ? 'active' : ''}`} onClick={handleSave} title="Save Changes" disabled={!isDirty}><FaSave /></button>
            <button className="icon-button-round" onClick={handleReset} title="초기화"><FaUndo /></button>
            {ocrLoading && (
              <div className="ocr-loading-overlay">
                <div className="ocr-loading-box">
                  <span>{ocrLoadingText}</span>
                  {/* 필요시 스피너 아이콘 등 추가 */}
                </div>
              </div>
            )}
            <span className="status-text">{status}</span>
          </div>
        </div>
      </header>
      <div className="content-area transactions-page">
        {/* 상단 툴바 */}
        <div className="transactions-toolbar card">
          <button onClick={handleAddRow}>
            {hasCheckedRows ? <FaArrowRight /> : <FaPlus />}
            {hasCheckedRows ? ' 행 삽입' : ' 행 추가'}
          </button>
          <button onClick={handleDeleteSelected}><FaTrash /> 행 삭제</button>
          <button onClick={handleClearAllFilters}><FaFilter /> 전체 필터 해제</button>
          <button onClick={handleScrollToBottom}><FaAngleDoubleDown /> 맨 아래로</button>
          <div className="divider"></div>
          <button onClick={handleApplyBold} disabled={!hasCheckedRows} title="굵게"><FaBold /></button>
          <button onClick={(e) => handleOpenColorPopup(e, 'flag')} disabled={!hasCheckedRows} title="플래그"><FaFlag /></button>
          <button onClick={(e) => handleOpenColorPopup(e, 'highlight')} disabled={!hasCheckedRows} title="형광펜"><FaHighlighter /></button>
          <button onClick={(e) => handleOpenColorPopup(e, 'background')} disabled={!hasCheckedRows} title="배경색"><FaFillDrip /></button>
          <div className="divider"></div>
          <button className="primary" onClick={() => setIsFormModalOpen(true)}>내역입력 폼 열기</button>
          <button className="primary" ref={ocrButtonRef} onClick={() => setOcrModalOpen(true)}>딥러닝 자동입력</button>
        </div>

        {/* 거래내역 테이블 */}
        <div className="table-container" ref={tableContainerRef}>
          <table>
            <thead>
              <tr>
                {/* TODO: 각 헤더에 필터 버튼 추가 */}
                <th>
                <input
                  type="checkbox"
                  onChange={handleToggleCheckAll}
                  // 보이는 행이 모두 체크되었을 때만 '전체 선택' 체크박스 활성화
                  checked={processedTransactions.length > 0 && processedTransactions.every(t => checkedRows.has(t.id))}
                />
              </th>
                {renderHeader('transaction_date', '날짜')}
                {renderHeader('account_name', '계좌')}
                {renderHeader('type', '유형')}
                {renderHeader('major_category_name', '대분류')}
                {renderHeader('minor_category_name', '소분류')}
                {renderHeader('amount', '금액')}
                {renderHeader('merchant', '거래처')}
                {renderHeader('memo', '메모')}
              </tr>
            </thead>
            <tbody>
              {processedTransactions.map((transaction) => {
                // 3가지 스타일 클래스를 모두 조합
                const classNames = [
                  transaction.is_bold ? 'bold-row' : '',
                  transaction.flag_color_id > 0 ? `flag-${transaction.flag_color_id}` : '',
                  transaction.highlight_color_id > 0 ? `highlight-${transaction.highlight_color_id}` : '',
                  transaction.background_color_id > 0 ? `bg-${transaction.background_color_id}` : '',
                ].filter(Boolean).join(' '); // 빈 문자열을 제거하고 공백으로 합침
                
                return(
                  <tr key={transaction.id} id={`row-${transaction.id}`} className={classNames.trim()}>
                  <td>
                  <input
                    type="checkbox"
                    checked={checkedRows.has(transaction.id)}
                    onChange={() => handleToggleCheck(transaction.id)}
                  />
                </td>
                  {renderCell(transaction, 'transaction_date')}
                  {renderCell(transaction, 'account_name')}
                  {renderCell(transaction, 'type')}
                  {renderCell(transaction, 'major_category_name')}
                  {renderCell(transaction, 'minor_category_name')}
                  {renderCell(transaction, 'amount')}
                  {renderCell(transaction, 'merchant')}
                  {renderCell(transaction, 'memo')}
                </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {/* OCR 모달 */}
        <OcrImageUploadModal
          isOpen={ocrModalOpen}
          onClose={() => setOcrModalOpen(false)}
          onUpload={handleOcrUpload}
          anchorRef={ocrButtonRef}
        />
        {/* 내역입력 폼 모달 */}
        <TransactionFormModal
          isOpen={isFormModalOpen}
          onClose={handleCloseFormModal}
          onInsert={handleInsertTransactions}
          appData={appData}
          allTransactions={originalTransactions}
          insertedCount={insertedCount}
          setInsertedCount={setInsertedCount}
        />
        {/* OCR 미리보기 팝업 */}
        <OcrPreviewTableModal
          open={ocrPreviewOpen}
          rows={ocrPreviewRows}
          onClose={() => setOcrPreviewOpen(false)}
          onInsert={handleOcrInsert}
          appData={appData}
          TRANSACTION_TYPES={TRANSACTION_TYPES}
        />
        {/* 플로팅 셀렉트 컴포넌트 */}
        <FloatingSelectPopup ref={floatingSelectRef} />
        {/* 필터 팝업 조건부 렌더링 */}
        {activeFilter && (
          <FilterPopup
            columnKey={activeFilter.column}
            columnName={activeFilter.name}
            allValues={transactions.map(t => t[activeFilter.column])}
            appliedFilters={filters[activeFilter.column] || []}
            onApply={handleApplyFilter}
            onClose={() => setActiveFilter(null)}
            onSort={handleSort}
            onClearFilter={handleClearColumnFilter}
            position={popupPosition}
          />
        )}
        {/* 확인/경고 팝업 조건부 렌더링 */}
        <ConfirmPopup
          isOpen={confirmPopup.isOpen}
          title={confirmPopup.title}
          message={confirmPopup.message}
          onConfirm={confirmPopup.onConfirm}
          onCancel={confirmPopup.onCancel}
          type={confirmPopup.type}
        />
        {/* 하이라이트 팝업 조건부 렌더링 */}
        {isColorPopupOpen && (
          <HighlightPopup
            position={colorPopupPosition}
            onSelectColor={handleApplyColor}
            onClose={() => setColorPopupOpen(false)}
            title={activeStyleType === 'flag' ? 'Flag' : activeStyleType === 'highlight' ? 'Highlight' : 'Background'}
          />
        )}
      </div>
    </div>
    </>
  );
};

export default Transactions;