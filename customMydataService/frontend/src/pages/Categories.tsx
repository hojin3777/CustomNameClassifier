import React, { useState, useEffect, useRef } from 'react';
import { useDirty } from '../App';
import { FaPlusCircle, FaPen, FaTrash, FaSave, FaUndo } from 'react-icons/fa';
import ConfirmPopup from '../components/ConfirmPopup';

// 타입 정의
type Account = {id: number; name: string};
type MinorCategory = { uuid: string; name: string };
type CategoryItem = { major: string; minors: MinorCategory[]};
type CategoriesData = CategoryItem[];
type EditingState = 
  | { type: 'account'; id: number}
  | { type: 'major'; major: string }
  | { type: 'minor'; major: string; minorIndex: number };
type AlertInfo = {
  isOpen: boolean;
  type?: 'input' | 'confirm' | 'alert' | 'destructive';
  message: string;
  onConfirm: (value?: string) => void;
  onCancel?: () => void;
  title?: string;
  placeholder?: string;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';
const Categories = () => {
  // vars
  const protectedCategories = ['고정수입', '유동수입', '이체분류'];
  const newAccountIdCounter = useRef(-1); // 새 계좌에 임시 음수 ID 부여용
  //states
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [categories, setCategories] = useState<CategoriesData>([]);
  const [status, setStatus] = useState('Loading...');
  const [editing, setEditing] = useState<EditingState | null>(null);
  const [originalEditValue, setOriginalEditValue] = useState<string>('');
  const [alertInfo, setAlertInfo] = useState<AlertInfo>({ isOpen: false, message: '', onConfirm: () => {} });
  const inputRef = useRef<HTMLInputElement>(null);
  // Dirty states
  const dirtyContext = useDirty();
  const isDirty = dirtyContext?.isDirty ?? false;
  const setIsDirty = dirtyContext?.setIsDirty ?? (() => {});
  

  // 경고 이벤트 리스너 등록
  useEffect(() => {
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      if (isDirty) {
        event.preventDefault();
        event.returnValue = '저장되지 않은 변경 사항이 있습니다. 정말 페이지를 떠나시겠습니까?';
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [isDirty]); // isDirty 상태가 변경될 때마다 이 효과를 다시 실행

  // ******* 데이터 로딩 및 저장 *******
  const fetchAllData = async () => {
    try {
      setStatus('Loading...');
      const accountsRes = await fetch(`${API_BASE_URL}/api/accounts`);
      const categoriesRes = await fetch(`${API_BASE_URL}/api/categories`);
      if (!accountsRes.ok || !categoriesRes.ok) {
        throw new Error('Failed to fetch data from server.');
      }

      const accountsData = await accountsRes.json();
      const rawCategoriesData = await categoriesRes.json();

      const processedCategories = rawCategoriesData.map((cat: any) => ({
        major: cat.major,
        minors: Array.isArray(cat.minors) ? cat.minors.map((minor: any) =>
          typeof minor === 'string' ? { uuid: '', name: minor } : minor
        ): [],
      }));

      setAccounts(accountsData); 
      setCategories(processedCategories);
      
      setIsDirty(false);
      setStatus('');
    } catch (error) {
      console.error("Failed to fetch Data:", error);
      setStatus('Failed to load Data.');
    }
  };

  useEffect(() => { fetchAllData(); }, []);
  useEffect(() => { if (editing && inputRef.current) inputRef.current.focus(); }, [editing]);

  // ******* 전체 데이터 저장 *******
  const handleSave = async () => {
    try {
      // 계좌 먼저 저장
      setStatus('Saving Accounts...');
      const accountsRes = await fetch(`${API_BASE_URL}/api/accounts`,{
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(accounts),
      });
      if (!accountsRes.ok) {
        const errorText = await accountsRes.text();
        console.error('Failed to save accounts:', errorText);
        throw new Error(`계좌 저장 실패: ${accountsRes.statusText}`);
      }
      const savedAccounts = await accountsRes.json();
      setAccounts(savedAccounts);

      // 계좌 저장 후 카테고리 저장
      setStatus('Saving Categories...');
      const categoriesRes = await fetch(`${API_BASE_URL}/api/categories`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(categories),
      });
      if (!categoriesRes.ok) {
        const errorText = await categoriesRes.text();
        console.error('Failed to save categories:', errorText);
        throw new Error(`카테고리 저장 실패: ${categoriesRes.statusText}`);
      }
      const savedCategories = await categoriesRes.json();
      const processedCategories = savedCategories.map((cat: any) => ({
        major: cat.major,
        minors: Array.isArray(cat.minors) ? cat.minors.map((minor: any) =>
          typeof minor === 'string' ? { uuid: '', name: minor } : minor
        ): [],
      }));
      setCategories(processedCategories);

      setIsDirty(false);
      setStatus('Saved successfully.');
      setTimeout(() => setStatus(''), 3000);
    } catch (error) {
      const message = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';
      console.error('Save failed:', error);
      setStatus('Save failed.')
      await fetchAllData(); // 저장 실패 시 데이터 다시 불러오기
    }
  };

  // ******* 편집 관련 함수 *******
  const startEditing = (state: EditingState) => {
    if (state.type === 'account') {
      const account = accounts.find(acc => acc.id === state.id);
      if (account){
        setOriginalEditValue(account.name);
      }
    } else if (state.type === 'major') {
      setOriginalEditValue(state.major);
    } else if (state.type === 'minor') {
      const category = categories.find(c => c.major === state.major);
      if (category) {
        setOriginalEditValue(category.minors[state.minorIndex]?.name);
      }
    }
    setEditing(state);
  };

  const handleEditKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleCategoryEditCommit((e.target as HTMLInputElement).value);
    } else if (e.key === 'Escape') {
      setEditing(null);
    }
  };

  const handleAccountEditKeyDown = (e: React.KeyboardEvent<HTMLInputElement>, id: number) => {
    if (e.key === 'Enter') {
      handleAccountEditCommit((e.target as HTMLInputElement).value, id);
    } else if (e.key === 'Escape') {
      setEditing(null);
    }
  };

  // ******* 계좌 관련 함수 *******
  const addAccount = () => {
    setAlertInfo({
      isOpen: true, type: 'input', message: '새로운 계좌 이름을 입력하세요:', placeholder: '계좌 이름 입력',
      onConfirm: (newName) => {
        if (newName && !accounts.some(acc => acc.name === newName)) {
          const newAccount: Account = { id: --newAccountIdCounter.current, name: newName };
          setAccounts(prev => [...prev, newAccount]);
          setIsDirty(true);
      }
        setAlertInfo({ ...alertInfo, isOpen: false });
      },
      onCancel: () => {
        setAlertInfo({ ...alertInfo, isOpen: false });
      }
    });
  };

  const deleteAccount = async (id: number) => {
    const account = accounts.find(acc => acc.id === id);
    if (!account) return;
    // 최소 1개의 계좌는 유지
    if(accounts.length <= 1) {
      setAlertInfo({
        isOpen: true, type: 'alert', message: '최소 1개의 계좌는 유지해야 합니다.',
        onConfirm: () => setAlertInfo({ ...alertInfo, isOpen: false }),
      });
      return;
    }
    // 계좌 사용 여부 확인(id가 양수인 경우에만 체크)
    if(id > 0){
      const usageCheckResponse = await fetch(`${API_BASE_URL}/api/accounts/usage?name=${encodeURIComponent(account.name)}`)
      const usageData = await usageCheckResponse.json();
      if (usageData.in_use) {
        setAlertInfo({
          isOpen: true,
          type: 'alert',
          message: `계좌 '${account.name}'는 거래내역에서 사용 중이므로 삭제할 수 없습니다.`,
          onConfirm: () => setAlertInfo({ ...alertInfo, isOpen: false }),
        });
        return;
      }
    }
    
    // 위 조건들을 통과한 경우에만 삭제 확인 팝업 표시
    setAlertInfo({
      isOpen: true, type: 'confirm', message: `'${account.name}' 계좌를 정말 삭제하시겠습니까?`,
      onConfirm: () => {
        setAccounts(prev => prev.filter(acc => acc.id !== id));
        setIsDirty(true);
        setAlertInfo({ ...alertInfo, isOpen: false });
      },
      onCancel: () => setAlertInfo({ ...alertInfo, isOpen: false }),
    });
  }

  const handleAccountEditCommit = (newValue: string, id: number) => {
    const trimmedValue = newValue.trim();
    const originalValue = accounts.find(acc => acc.id === id);
    if(trimmedValue && originalValue && trimmedValue !== originalValue.name && !accounts.some(acc => acc.name === trimmedValue)){
      setAccounts(prev => prev.map(acc => acc.id === id ? { ...acc, name: trimmedValue } : acc));
      setIsDirty(true);
    }
    setEditing(null);
  };

  const renderAccountItem = (account: Account) => {
    const isEditing = editing?.type === 'account' && editing.id === account.id;
    return isEditing ? (
      <input
        key={`editing-${account.id}`}
        ref={inputRef}
        type="text"
        defaultValue={account.name}
        onBlur={(e) => handleAccountEditCommit(e.target.value, account.id)}
        onKeyDown={(e) => handleAccountEditKeyDown(e, account.id)}
        className="inline-edit-input"
      />
    ) : (
      <div key={account.id} className='category-item' onDoubleClick={() => startEditing({ type: 'account', id: account.id })}>
        <span>{account.name}</span>
        <div className="item-actions">
          <FaPen className="action-icon edit" onClick={() => setEditing({ type: 'account', id: account.id })} />
          <FaTrash className="action-icon delete" onClick={() => deleteAccount(account.id)} />
        </div>
      </div>
    );
  }

  // ******* 카테고리 관련 함수 *******
  const handleReset = () => {
    setAlertInfo({
      isOpen: true,
      type: 'destructive',
      title: '카테고리 초기화 경고',
      message: '모든 카테고리를 기본값으로 초기화하시겠습니까? 이 작업은 되돌릴 수 없습니다.',
      onConfirm: async () => {
        setAlertInfo({ ...alertInfo, isOpen: false });
        setStatus('Resetting...');
        await fetch(`${API_BASE_URL}/api/categories`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify([]) });
        await fetchAllData();
        setIsDirty(true);
        setStatus('Reset successfully!');
        setTimeout(() => setStatus(''), 3000);
      },
      onCancel: () => setAlertInfo({ ...alertInfo, isOpen: false }),
    });
  };

  const addMajorCategory = () => {
    if (categories.length >= 30) return;
    setAlertInfo({
      isOpen: true,
      type: 'input',
      message: '새로운 대분류 이름을 입력하세요:',
      placeholder: '대분류 이름 입력',
      onConfirm: (newMajor) => {
        if (newMajor && !categories.find(c => c.major === newMajor)) {
          setCategories(prev => [...prev, { major: newMajor, minors: [] }]);
          setIsDirty(true);
        }
        setAlertInfo({ ...alertInfo, isOpen: false });
      },
      onCancel: () => setAlertInfo({ ...alertInfo, isOpen: false }),
    });
  }

  const addMinorCategory = (major: string) => {
    const category = categories.find(c => c.major === major);
    if (!category || category.minors.length >= 20) return;
    setAlertInfo({
      isOpen: true,
      type: 'input',
      message: `${major}에 추가할 소분류 이름을 입력하세요:`,
      placeholder: '소분류 이름 입력',
      onConfirm: (newMinorName) => {
        if (newMinorName) {
          const newMinor: MinorCategory = { name: newMinorName, uuid: '' };
          setCategories(prev => prev.map(c =>
            c.major === major ? { ...c, minors: [...c.minors, newMinor] } : c
          ));
          setIsDirty(true);
        }
        setAlertInfo({ ...alertInfo, isOpen: false });
      },
      onCancel: () => setAlertInfo({ ...alertInfo, isOpen: false }),
    });
  };

  const deleteMajorCategory = (major: string) => {
    if (protectedCategories.includes(major)) return;
    setAlertInfo({
      isOpen: true,
      type: 'destructive',
      title: '대분류 삭제',
      message: `'${major}' 대분류와 모든 하위 항목을 삭제하시겠습니까?`,
      onConfirm: () => {
        setCategories(prev => prev.filter(c => c.major !== major));
        setIsDirty(true);
        setAlertInfo({ ...alertInfo, isOpen: false });
      },
      onCancel: () => setAlertInfo({ ...alertInfo, isOpen: false }),
    });
  };

  const deleteMinorCategory = async (major: string, minor: MinorCategory) => {
    if(minor.uuid){
      const usageCheckResponse = await fetch(`${API_BASE_URL}/api/categories/usage?uuid=${minor.uuid}`);
      const usageData = await usageCheckResponse.json();
      if (usageData.in_use) {
        setAlertInfo({
          isOpen: true,
          type: 'alert',
          message: `카테고리 '${major}-${minor.name}'는 거래내역에서 사용 중이므로 삭제할 수 없습니다.`,
          onConfirm: () => setAlertInfo({ ...alertInfo, isOpen: false }),
        });
        return;
      }
    }
    setAlertInfo({
      isOpen: true,
      type: 'confirm',
      message: `'${minor.name}' 항목을 정말 삭제하시겠습니까?`,
      onConfirm: () => {
        setCategories(prev => prev.map(c => {
          if (c.major !== major) return c;
          const updatedMinors = c.minors.filter(m => (m as MinorCategory).uuid ? (m as MinorCategory).uuid !== minor.uuid : (m as MinorCategory).name !== minor.name);
          return { ...c, minors: updatedMinors };
      }));
        setIsDirty(true);
        setAlertInfo({ ...alertInfo, isOpen: false });
      },
      onCancel: () => setAlertInfo({ ...alertInfo, isOpen: false }),
    });
  };

  const handleCategoryEditCommit = (newValue: string) => {
    if (!editing || !newValue) {
      setEditing(null);
      return;
    }
    let isChanged = false;
    if (editing.type === 'major') {
      const originalValue = editing.major;
      if (newValue !== originalValue && !categories.some(c => c.major === newValue)) {
        setCategories(prev => prev.map(c => c.major === originalValue ? { ...c, major: newValue } : c));
        isChanged = true;
      }
    } else if (editing.type === 'minor') {
      const originalValue = categories.find(c => c.major === editing.major)?.minors[editing.minorIndex!].name;
      if (newValue !== originalValue) {
        setCategories(prev => prev.map(c => {
          if (c.major !== editing.major) return c;
          const newMinors = c.minors.map((m, i) => i === editing.minorIndex ? { ...m, name: newValue } : m);
          return { ...c, minors: newMinors };
        }));
        isChanged = true;
      }
    }
    
    if (isChanged) {
      setIsDirty(true); // ✨ isDirty 직접 설정
    }
    setEditing(null);
  };

  const renderItem = (major: string, minor: MinorCategory, index: number) => {
    const isEditing = editing?.type === 'minor' && editing.major === major && editing.minorIndex === index;
    return isEditing ? (
      <input
        ref={inputRef}
        type="text"
        defaultValue={minor.name}
        onBlur={(e) => handleCategoryEditCommit(e.target.value)}
        onKeyDown={handleEditKeyDown}
        className="inline-edit-input"
      />
    ) : (
      <div key={minor.uuid || `new-${index}`} className="category-item" onDoubleClick={() => startEditing({ type: 'minor', major, minorIndex: index })}>
        <span>{minor.name}</span>
        <div className="item-actions">
          <FaPen className="action-icon edit" onClick={() => setEditing({ type: 'minor', major, minorIndex: index })} />
          <FaTrash className="action-icon delete" onClick={() => deleteMinorCategory(major, minor)} />
        </div>
      </div>
    );
  };

  const renderHeader = (major: string) => {
    const isEditing = editing?.type === 'major' && editing.major === major;
    const isProtected = protectedCategories.includes(major);
    return isEditing ? (
      <input
        ref={inputRef}
        type="text"
        defaultValue={major}
        onBlur={(e) => handleCategoryEditCommit(e.target.value)}
        onKeyDown={handleEditKeyDown}
        className='inline-edit-input header-edit'
      />
    ) : (
      <div className="card category-header" onDoubleClick={() => !isProtected && startEditing({ type: 'major', major })}>
        <span>{major}</span>
        {!isProtected && (
          <div className="item-actions">
            <FaPen className="action-icon edit" onClick={() => setEditing({ type: 'major', major })} />
            <FaTrash className="action-icon delete" onClick={() => deleteMajorCategory(major)} />
          </div>
        )}
      </div>
    );
  };

  return (
    <div className='categories-page-wrapper'>      
      <header className="main-header">
        <div className="header-title-group">
          <h1>Categories</h1>
          <div className="header-actions">
            <span className="status-text">{status}</span>
            <button className={`icon-button-round ${isDirty ? 'active' : ''}`} onClick={handleSave} title="Save Changes" disabled={!isDirty}><FaSave /></button>
            <button className="icon-button-round" onClick={handleReset} title="Reset to Default"><FaUndo /></button>
          </div>
        </div>
      </header>
      <div className="content-area categories-page">
        <div className="categories-grid-container">
          <div className="categories-grid">
            {/* Accounts Column */}
            <div className="category-column-wrapper">
              <div className='card category-header'>계좌</div>
              <div className="card category-column">
                {accounts.map((account) => renderAccountItem(account))}
                {accounts.length < 20 && (
                  <div className="category-item add-item" onClick={addAccount}>
                    <FaPlusCircle />
                  </div>
                )}
              </div>
            </div>
            <div className="vertical-divider"></div>
            {/* Categories Columns */}
            {categories.map(({ major, minors }) => (
              <div className="category-column-wrapper" key={major}>
              {renderHeader(major)}
              <div className="card category-column">
                {minors.map((minor, index) => renderItem(major, minor as MinorCategory, index))}
                {minors.length < 20 && (
                  <div className="category-item add-item" onClick={() => addMinorCategory(major)}>
                    <FaPlusCircle />
                  </div>
                )}
              </div>
            </div>
          ))}
          {/* Add Major Category Button */}
          {categories.length < 27 && (
            <div className="category-column-wrapper add-major-wrapper" onClick={addMajorCategory}>
              <div className="card add-major-card">
                <FaPlusCircle />
                <span>Add Major Category</span>
              </div>
            </div>
          )}
          </div>
        </div>
      </div>
      <ConfirmPopup {...alertInfo} />
    </div>
  );
};


export default Categories;