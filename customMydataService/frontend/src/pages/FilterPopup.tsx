import React, { useState, useEffect, useMemo } from 'react';
import { FaSortAlphaDown, FaSortAlphaUp, FaFilter, FaSortAmountUp } from 'react-icons/fa';
import './FilterPopup.css';
import type { Transaction } from './Transactions'; // ✨ Transaction 타입 import

type FilterPopupProps = {
  columnKey: string;
  columnName: string;
  allValues: any[];
  appliedFilters: any[];
  onApply: (columnKey: string, selectedValues: any[]) => void;
  onClose: () => void;
  // ✨ 1. 정렬, 필터해제, 위치 props 추가
  onSort: (columnKey: keyof Transaction, direction: 'asc' | 'desc') => void; // ✨ 2. 타입 명시
  onClearFilter: (columnKey:string) => void;
  position: { top: number; left: number };
};

const FilterPopup: React.FC<FilterPopupProps> = ({
  columnKey,
  columnName,
  allValues,
  appliedFilters,
  onApply,
  onClose,
  onSort,
  onClearFilter,
  position,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedValues, setSelectedValues] = useState<any[]>(appliedFilters);

  const uniqueValues = useMemo(() => Array.from(new Set(allValues)).sort(), [allValues]);

  const filteredOptions = useMemo(() => {
    return uniqueValues.filter(value =>
      value.toString().toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [uniqueValues, searchTerm]);

  const handleSelectAll = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.checked) {
      setSelectedValues(uniqueValues);
    } else {
      setSelectedValues([]);
    }
  };

  const handleValueChange = (value: any) => {
    setSelectedValues(prev =>
      prev.includes(value) ? prev.filter(v => v !== value) : [...prev, value]
    );
  };

  const handleApplyClick = () => {
    onApply(columnKey, selectedValues);
    onClose();
  };

  const handleClearAndClose = () => {
    onClearFilter(columnKey);
    onClose();
  };

  return (
    <div className="filter-popup-overlay" onClick={onClose}>
      <div className="filter-popup-content" onClick={(e) => e.stopPropagation()} style={position}>
        <div className="filter-popup-header">
          {/* ✨ 4. 정렬 및 필터 해제 onClick 이벤트 연결 */}
          <button title="오름차순 정렬" onClick={() => onSort(columnKey as keyof Transaction, 'asc')}><FaSortAlphaUp /></button>
          <button title="내림차순 정렬" onClick={() => onSort(columnKey as keyof Transaction, 'desc')}><FaSortAlphaDown /></button>
          <button title="필터 해제" onClick={handleClearAndClose}><FaFilter /></button>
        </div>
        <div className="filter-popup-body">
          <input
            type="text"
            className="filter-search-input"
            placeholder="목록 검색..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <div className="filter-options-list">
            <label>
              <input
                type="checkbox"
                checked={selectedValues.length === uniqueValues.length}
                onChange={handleSelectAll}
              />
              (모두 선택)
            </label>
            {filteredOptions.map((value, index) => (
              <label key={index}>
                <input
                  type="checkbox"
                  checked={selectedValues.includes(value)}
                  onChange={() => handleValueChange(value)}
                />
                {value.toString() || '(비어 있음)'}
              </label>
            ))}
          </div>
        </div>
        <div className="filter-popup-footer">
          <button onClick={handleApplyClick} className="primary">확인</button>
          <button onClick={onClose}>취소</button>
        </div>
      </div>
    </div>
  );
};

export default FilterPopup;