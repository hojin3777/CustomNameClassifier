import React, { useState, useEffect } from 'react';
import { FaTimes, FaWindowMaximize, FaMinus } from 'react-icons/fa';
import './TitleBar.css';

const TitleBar: React.FC = () => {
  const [iconPath, setIconPath] = useState<string>('');
  useEffect(() => {
    // Electron 환경 확인
    const isDev = window.location.protocol === 'http:';
    const isMac = navigator.platform.toLowerCase().includes('mac');
    
    if (isDev) {
      // 개발 모드: Vite dev server
      setIconPath(isMac ? '/icon.icns' : '/icon.ico');
    } else {
      // 프로덕션 모드: file:// 프로토콜
      // Electron에서 app.asar 내부 또는 resources 폴더의 플랫폼별 아이콘 사용
      setIconPath(isMac ? './icon.icns' : './icon.ico');
    }
  }, []);

  const handleMinimize = () => {
    window.electronAPI?.minimize();
  };

  const handleMaximize = () => {
    window.electronAPI?.maximize();
  };

  const handleClose = () => {
    window.electronAPI?.close();
  };

  return (
    <div className="titlebar">
      <div className="titlebar-drag-region">
        <div className="titlebar-title">
          {iconPath && (
            <img 
              src={iconPath}
              alt="logo" 
              className="titlebar-icon"
              onError={(e) => {
                console.error('Icon failed to load:', iconPath);
                e.currentTarget.style.display = 'none';
              }}
            />
          )}
          <span>개인화 마이데이터 서비스</span>
        </div>
      </div>
      
      <div className="titlebar-controls">
        <button className="titlebar-button minimize" onClick={handleMinimize} title='최소화'>
          <FaMinus />
        </button>
        <button className="titlebar-button maximize" onClick={handleMaximize} title='최대화'>
          <FaWindowMaximize />
        </button>
        <button className="titlebar-button close" onClick={handleClose} title='닫기'>
          <FaTimes />
        </button>
      </div>
    </div>
  );
};

export default TitleBar;