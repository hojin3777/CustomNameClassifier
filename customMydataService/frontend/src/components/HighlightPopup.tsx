import React, { useEffect, useRef } from 'react';
import './HighlightPopup.css';

// 색상 팔레트 정의 (id: 0은 '없음')
export const HIGHLIGHT_COLORS = [
  { id: 0, color: 'transparent', name: '없음' },
  { id: 1, color: '#FFF38A', name: '노랑' },
  { id: 2, color: '#AEE4FF', name: '하늘' },
  { id: 3, color: '#FFC1D6', name: '핑크' },
  { id: 4, color: '#B8F3B8', name: '연두' },
  { id: 5, color: '#FFDDAA', name: '주황' },
  { id: 6, color: '#E0C8FF', name: '보라' },
];

type HighlightPopupProps = {
  position: { top: number; left: number };
  onSelectColor: (colorId: number) => void;
  onClose: () => void;
};

const HighlightPopup: React.FC<HighlightPopupProps> = ({ position, onSelectColor, onClose }) => {
  const popupRef = useRef<HTMLDivElement>(null);

  // 외부 클릭 시 닫기
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (popupRef.current && !popupRef.current.contains(event.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [onClose]);

  return (
    <div
      ref={popupRef}
      className="highlight-popup"
      style={{ top: position.top, left: position.left }}
    >
      {HIGHLIGHT_COLORS.map(({ id, color, name }) => (
        <button
          key={id}
          className="color-swatch-btn"
          title={name}
          onClick={() => onSelectColor(id)}
        >
          {id === 0 ? (
            <div className="color-swatch none">
              <svg viewBox="0 0 24 24">
                <line x1="4" y1="20" x2="20" y2="4" stroke="red" strokeWidth="2" />
              </svg>
            </div>
          ) : (
            <div className="color-swatch" style={{ backgroundColor: color }}></div>
          )}
        </button>
      ))}
    </div>
  );
};

export default HighlightPopup;