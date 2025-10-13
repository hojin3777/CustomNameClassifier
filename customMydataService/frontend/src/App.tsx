import React, { useState, createContext, useContext, useEffect } from 'react';
import { HashRouter, Routes, Route, NavLink, Navigate, useNavigate } from 'react-router-dom'; //BrowserRouter,
import './App.css';
import TitleBar from './TitleBar';
import Dashboard from './pages/Dashboard';
import Monthly from './pages/Monthly';
import Transactions from './pages/Transactions';
import Categories from './pages/Categories';
import Mapping from './pages/Mapping';
import { FaBars, FaQuestionCircle, FaCog, FaHourglassStart, FaHourglassHalf, FaHourglassEnd } from 'react-icons/fa';

// isDirty 상태를 전역적으로 관리하기 위한 Context 생성
const DirtyContext = createContext<{ isDirty: boolean; setIsDirty: (dirty: boolean) => void; } | null>(null);
export const useDirty = () => useContext(DirtyContext);

// isDirty 상태를 확인하고 내비게이션을 처리하는 NavLink 래퍼 컴포넌트
const GuardedNavLink = ({ to, children }: { to: string; children: React.ReactNode }) => {
  const dirtyContext = useDirty();
  const navigate = useNavigate();

  const handleClick = (event: React.MouseEvent<HTMLAnchorElement>) => {
    event.preventDefault(); // 일단 기본 이동을 막습니다.
    if (dirtyContext?.isDirty) {
      if (window.confirm('저장하지 않은 변경사항이 있습니다. 페이지를 떠나시겠습니까?')) {
        dirtyContext.setIsDirty(false); // 상태를 초기화하고
        navigate(to); // 수동으로 이동합니다.
      }
      // 사용자가 '취소'를 누르면 아무것도 하지 않습니다.
    } else {
      navigate(to); // 저장할 내용이 없으면 바로 이동합니다.
    }
  };

  return <NavLink to={to} onClick={handleClick}>{children}</NavLink>;
};

const AppContent = () => {
  const [isSidebarOpen, setSidebarOpen] = useState(false);
  const openSidebar = () => setSidebarOpen(true);
  const closeSidebar = () => setSidebarOpen(false);

  return (
    <>
      <TitleBar />
      <div className="app-wrapper">
        <div className={`sidebar-container ${isSidebarOpen ? 'open' : ''}`} onMouseLeave={closeSidebar}>
          <div className={`sidebar-wrapper ${isSidebarOpen ? 'open' : ''}`}>
            <aside className="sidebar">
              <div className="sidebar-header"><h2>Custom MyData</h2></div>
              <nav className="sidebar-nav">
                <ul>
                  <li><GuardedNavLink to="/dashboard">Dashboard</GuardedNavLink></li>
                  <li><GuardedNavLink to="/monthly">Monthly</GuardedNavLink></li>
                  <li><GuardedNavLink to="/transactions">Transactions</GuardedNavLink></li>
                  <li><GuardedNavLink to="/categories">Categories</GuardedNavLink></li>
                  <li><GuardedNavLink to="/mapping">Mapping</GuardedNavLink></li>
                </ul>
              </nav>
              <div className="sidebar-footer">
                <button className="icon-button"><FaQuestionCircle /></button>
                <button className="icon-button"><FaCog /></button>
              </div>
            </aside>
          </div>
        </div>
        <main className="main-content">
          <FaBars className="hamburger-menu" onMouseEnter={openSidebar} />
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/monthly" element={<Monthly />} />
            <Route path="/transactions" element={<Transactions />} />
            <Route path="/categories" element={<Categories />} />
            <Route path="/mapping" element={<Mapping />} />
          </Routes>
        </main>
      </div>
    </>
  );
};

const App = () => {
  const [isDirty, setIsDirty] = useState(false);
  const [backendReady, setBackendReady] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [hourglassIcon, setHourglassIcon] = useState(0);

  useEffect(() => {
    if (!backendReady) {
      const iconInterval = setInterval(() => {
        setHourglassIcon(prev => (prev + 1) % 3); // 0 -> 1 -> 2 -> 0 순환
      }, 1000);

      return () => clearInterval(iconInterval);
    }
  }, [backendReady]);

  useEffect(() => {
    let isMounted = true;
    const checkBackend = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/health', {
          method: 'GET',
          headers: { 'Accept': 'application/json' }
        });

        if (response.ok && isMounted) {
          console.log('Backend is ready!');
          setBackendReady(true);
        } else if (isMounted) {
          throw new Error('Backend not ready');
        }
      } catch (error) {
        if (isMounted) {
          console.log(`Backend not ready yet (attempt ${retryCount + 1})...`);
          setRetryCount(prev => prev + 1);

          // 최대 180번 재시도 (3분)
          if (retryCount < 180) {
            setTimeout(checkBackend, 1000); // 1초 후 재시도
          } else {
            console.error('Backend failed to start after 180 attempts');
            alert('백엔드 서버를 시작할 수 없습니다. 앱을 다시 시작해주세요.');
          }
        }
      }
    };
    checkBackend();
    return () => {
      isMounted = false; // 클린업 시 플래그 해제
    };
  }, [retryCount]); // retryCount 의존성 추가

  // getHourglassIcon: 현재 시계 아이콘 반환
  const getHourglassIcon = () => {
    switch (hourglassIcon) {
      case 0:
        return <FaHourglassStart />;
      case 1:
        return <FaHourglassHalf />;
      case 2:
        return <FaHourglassEnd />;
      default:
        return <FaHourglassStart />;
    }
  };

  if (!backendReady) {
    return (
      <>
        <TitleBar />
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100vh',
          backgroundColor: 'var(--color-bg-content)',
          color: 'var(--color-text-primary)',
          fontSize: '18px',
          flexDirection: 'column'
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{
              marginBottom: '24px',
              fontSize: '64px',
              transition: 'transform 0.3s ease' // 부드러운 전환
            }}>
              {getHourglassIcon()}
            </div>
            <div style={{ fontSize: '20px', fontWeight: 600, marginBottom: '12px' }}>
              백엔드 서버를 시작하는 중...
            </div>
            <div style={{ fontSize: '14px', color: 'var(--color-text-secondary)', marginBottom: '8px' }}>
              최초 실행 시 시간이 소요될 수 있습니다.
            </div>
            <div style={{ fontSize: '13px', color: 'var(--color-text-tertiary)' }}>
              재시도 {retryCount}/180
            </div>
          </div>
        </div>
      </>
    );
  }
  return (
    <HashRouter>
      <DirtyContext.Provider value={{ isDirty, setIsDirty }}>
        <AppContent />
      </DirtyContext.Provider>
    </HashRouter>
  );
};

export default App;