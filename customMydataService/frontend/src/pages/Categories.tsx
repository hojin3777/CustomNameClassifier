import React, { useState, useEffect, useRef } from 'react';
import { useDirty } from '../App';
import { FaPlusCircle, FaPen, FaTrash, FaSave, FaUndo } from 'react-icons/fa';

// 타입 정의
type MinorCategory = { uuid: string; name: string };
type CategoryItem = { major: string; minors: (MinorCategory | string)[]};
type CategoriesData = CategoryItem[];
type EditingState = { type: 'major' | 'minor'; major: string; minorIndex?: number; };

// 커스텀 입력 팝업
const CustomPrompt: React.FC<{ title: string; onConfirm: (value: string) => void; onCancel: () => void; }> = ({ title, onConfirm, onCancel }) => {
  const [value, setValue] = useState('');
  return (
    <div className="custom-alert-overlay">
      <div className="custom-alert-box">
        <p>{title}</p>
        <input type="text" value={value} onChange={(e) => setValue(e.target.value)} autoFocus onKeyDown={(e) => e.key === 'Enter' && onConfirm(value)} />
        <div className="custom-alert-buttons">
          <button onClick={() => onConfirm(value)} className="confirm">확인</button>
          <button onClick={onCancel}>취소</button>
        </div>
      </div>
    </div>
  );
};

// 커스텀 확인 팝업
const CustomAlert: React.FC<{ message: string; onConfirm: () => void; onCancel: () => void; }> = ({ message, onConfirm, onCancel }) => (
  <div className="custom-alert-overlay">
    <div className="custom-alert-box">
      <p>{message}</p>
      <div className="custom-alert-buttons">
        <button onClick={onConfirm} className="confirm">확인</button>
        <button onClick={onCancel}>취소</button>
      </div>
    </div>
  </div>
);

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';
const Categories = () => {
  const [categories, setCategories] = useState<CategoriesData>([]);
  const [status, setStatus] = useState('Loading...');
  const [editing, setEditing] = useState<EditingState | null>(null);
  const [alertInfo, setAlertInfo] = useState<{ 
    type: 'alert' | 'prompt'; 
    message: string; 
    onConfirm: (value?: any) => void; 
    onCancel?: () => void; 
  } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const dirtyContext = useDirty();
  const isDirty = dirtyContext?.isDirty ?? false;
  const setIsDirty = dirtyContext?.setIsDirty ?? (() => {});
  const protectedCategories = ['계좌', '고정수입', '유동수입', '이체분류'];

  // ✨ 4. 경고 이벤트 리스너 등록
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

  const fetchCategories = async () => {
    try {
      setStatus('Loading...');
      const response = await fetch(`${API_BASE_URL}/api/categories`);
      if (!response.ok) throw new Error('Network response was not ok.');
      const data: CategoriesData = await response.json();
      // ✨ 4. 데이터를 불러올 때 두 상태를 모두 업데이트
      setCategories(data);
      setIsDirty(false);
      setStatus('');
    } catch (error) {
      console.error("Failed to fetch categories:", error);
      setStatus('Failed to load categories.');
    }
  };

  useEffect(() => { fetchCategories(); }, []);
  useEffect(() => { if (editing && inputRef.current) inputRef.current.focus(); }, [editing]);

  // '저장' 버튼을 눌렀을 때만 모든 변경사항을 서버에 전송합니다.
  const handleSave = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/categories`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // ✨ 백엔드 save_categories_to_db는 이제 소분류가 객체인 데이터를 기대합니다.
        body: JSON.stringify(categories),
      });

      // ✨ 저장 후 백엔드가 반환하는 데이터도 uuid가 포함된 새로운 형식입니다.
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error || '저장에 실패했습니다.');
      }
      
      setCategories(result);
      setIsDirty(false);
      alert('성공적으로 저장되었습니다.');

    } catch (error) {
      let message = '알 수 없는 오류가 발생했습니다.';
      if (error instanceof Error) message = error.message;
      console.error('Save failed:', error);
      alert(`저장 실패: ${message}`);
      // 실패 시, DB와 상태를 동기화하기 위해 데이터를 다시 불러옵니다.
      await fetchCategories();
    }
  };

  const handleReset = () => {
    setAlertInfo({
      type: 'alert',
      message: '모든 카테고리를 기본값으로 초기화하시겠습니까?',
      onConfirm: async () => {
        setStatus('Resetting...');
        await fetch(`${API_BASE_URL}/api/categories`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify([]) });
        await fetchCategories();
        setIsDirty(true);
        setStatus('Reset successfully!');
        setAlertInfo(null);
        setTimeout(() => setStatus(''), 3000);
      }
    });
  };

  const addMajorCategory = () => {
    if (categories.length >= 30) return;
    setAlertInfo({
      type: 'prompt',
      message: '새로운 대분류 이름을 입력하세요:',
      onConfirm: (newMajor) => {
        if (newMajor && !categories.find(c => c.major === newMajor)) {
          setCategories(prev => [...prev, { major: newMajor, minors: [] }]);
          setIsDirty(true);
        }
        setAlertInfo(null);
      }
    });
  };

  const addMinorCategory = (major: string) => {
    const category = categories.find(c => c.major === major);
    if (!category || category.minors.length >= 20) return;
    setAlertInfo({
      type: 'prompt',
      message: `${major}에 추가할 소분류 이름을 입력하세요:`,
      onConfirm: (newMinorName) => {
        if (newMinorName) {
          // ✨ 새 소분류 객체 생성 (uuid는 백엔드에서 생성하므로 여기서는 빈 값)
          const newMinor: MinorCategory = { name: newMinorName, uuid: '' };
          setCategories(prev => prev.map(c => {
            if (c.major !== major) return c;
            // ✨ 2. 타입스크립트가 타입을 추론할 수 있도록 명확하게 작성
            const updatedMinors = [...c.minors, newMinor];
            return { ...c, minors: updatedMinors };
          }));
          setIsDirty(true);
        }
        setAlertInfo(null);
      }
    });
  };

  const deleteMajorCategory = (major: string) => {
    if (protectedCategories.includes(major)) return;
    setAlertInfo({
      type: 'alert',
      message: `'${major}' 대분류와 모든 하위 항목을 삭제하시겠습니까?`,
      onConfirm: () => {
        setCategories(prev => prev.filter(c => c.major !== major));
        setIsDirty(true);
        setAlertInfo(null);
      }
    });
  };

  const deleteItem = async (major: string, minor: MinorCategory) => {
    // ✨ 2. 삭제할 아이템의 uuid를 사용
    try {
      const usageCheckResponse = await fetch(`${API_BASE_URL}/api/categories/usage?uuid=${minor.uuid}`);
      const usageData = await usageCheckResponse.json();

      if (usageData.in_use) {
        alert(`카테고리 '${major}/${minor.name}'는 거래내역에서 사용 중이므로 삭제할 수 없습니다.`);
        return;
      }

      setAlertInfo({
        type: 'alert',
        message: `'${minor.name}' 항목을 정말 삭제하시겠습니까?`,
        onConfirm: () => {
          // ✨ uuid를 기준으로 필터링하여 삭제
          setCategories(prev => prev.map(c => {
            if (c.major !== major) return c;
            // ✨ 3. filter의 결과가 올바른 타입이 되도록 명시적 타입 단언 사용
            const updatedMinors = c.minors.filter(m => {
              if (typeof m === 'string') return true; // 계좌 항목은 항상 유지
              return m.uuid !== minor.uuid;
            });
            return { ...c, minors: updatedMinors };
          }));
          setIsDirty(true);
          setAlertInfo(null);
        },
        onCancel: () => setAlertInfo(null)
      });

    } catch (error) {
      console.error("Failed to check category usage:", error);
      alert('삭제 처리 중 오류가 발생했습니다.');
    }
  };

  const handleEditCommit = (e: React.FocusEvent<HTMLInputElement> | React.KeyboardEvent<HTMLInputElement>) => {
    if (!editing) return;
    const newValue = (e.target as HTMLInputElement).value.trim();
    const { type, major, minorIndex } = editing;
    
    const originalValue = (type === 'major') 
      ? major 
      : (major === '계좌'
          ? categories.find(c => c.major === major)?.minors[minorIndex!]
          : (categories.find(c => c.major === major)?.minors[minorIndex!] as MinorCategory).name);

    setEditing(null);

    if (!newValue || newValue === originalValue) return;

    if (type === 'major') {
      setCategories(prev => prev.map(c => c.major === major ? { ...c, major: newValue } : c));
    } else { // type === 'minor'
      setCategories(prev => prev.map(c => {
        if (c.major !== major) return c;
        // ✨ 4. map의 결과가 올바른 타입이 되도록 명시적 타입 단언 사용
        const updatedMinors = c.minors.map((m, i) => {
          if (i !== minorIndex) return m;
          return typeof m === 'string' ? newValue : { ...m, name: newValue };
        });
        return { ...c, minors: updatedMinors };
      }));
    }
    setIsDirty(true);
  };

  const renderItem = (major: string, minor: MinorCategory | string, index: number) => {
    const isAccount = typeof minor === 'string';
    const minorName = isAccount ? minor : minor.name;
    const uniqueKey = isAccount ? `account-${minorName}-${index}` : minor.uuid || `new-item-${index}`;
    const isEditing = editing?.type === 'minor' && editing.major === major && editing.minorIndex === index;
    return isEditing ? (
      <input
        ref={inputRef}
        type="text"
        defaultValue={minorName} // ✨ .name 사용
        onBlur={handleEditCommit}
        onKeyDown={(e) => e.key === 'Enter' && handleEditCommit(e)}
        className="inline-edit-input"
      />
    ) : (
      <div key={uniqueKey} className="category-item" onDoubleClick={() => setEditing({ type: 'minor', major, minorIndex: index })}>
        <span>{minorName}</span>
        <div className="item-actions">
          <FaPen className="action-icon edit" onClick={() => setEditing({ type: 'minor', major, minorIndex: index })} />
          {/* 계좌 항목은 삭제 불가 */}
          {!isAccount && <FaTrash className="action-icon delete" onClick={() => deleteItem(major, minor as MinorCategory)} />}
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
        onBlur={handleEditCommit}
        onKeyDown={(e) => e.key === 'Enter' && handleEditCommit(e)}
        className="inline-edit-input header-edit"
      />
    ) : (
      <div className="card category-header" onDoubleClick={() => !isProtected && setEditing({ type: 'major', major })}>
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
    <>
      {alertInfo?.type === 'alert' && <CustomAlert message={alertInfo.message} onConfirm={alertInfo.onConfirm} onCancel={() => setAlertInfo(null)} />}
      {alertInfo?.type === 'prompt' && <CustomPrompt title={alertInfo.message} onConfirm={alertInfo.onConfirm} onCancel={() => setAlertInfo(null)} />}
      
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
            {categories.map(({ major, minors }) => (
              <React.Fragment key={major}>
                <div className="category-column-wrapper">
                  {renderHeader(major)}
                  <div className="card category-column">
                    {minors.map((minor, index) => renderItem(major, minor, index))}
                    {minors.length < 20 && (
                      <div className="category-item add-item" onClick={() => addMinorCategory(major)}>
                        <FaPlusCircle />
                      </div>
                    )}
                  </div>
                </div>
                {major === '계좌' && <div className="vertical-divider"></div>}
              </React.Fragment>
            ))}
            {categories.length < 30 && (
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
    </>
  );
};

export default Categories;