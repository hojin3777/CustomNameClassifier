import React from 'react';
import './ConfirmPopup.css';

type ConfirmPopupProps = {
  isOpen: boolean;
  title: string;
  message: string;
  onConfirm: () => void;
  onCancel?: () => void;
  // ✨ 1. 팝업 타입을 결정하는 prop 추가 ('destructive'는 빨간 버튼)
  type?: 'info' | 'destructive';
};

const ConfirmPopup: React.FC<ConfirmPopupProps> = ({
  isOpen,
  title,
  message,
  onConfirm,
  onCancel,
  type = 'info', // 기본값은 'info'
}) => {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="confirm-popup-overlay">
      <div className="confirm-popup-content">
        <div className="confirm-popup-header">
          <h3>{title}</h3>
        </div>
        <div className="confirm-popup-body">
          <p>{message}</p>
        </div>
        <div className="confirm-popup-footer">
          {onCancel && (
            <button onClick={onCancel} className="secondary">
              취소
            </button>
          )}
          {/* ✨ 2. type에 따라 버튼 클래스를 동적으로 부여 */}
          <button onClick={onConfirm} className={type === 'destructive' ? 'destructive' : 'primary'}>
            확인
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmPopup;