import React, { useState, createContext, useContext } from 'react';
import { BrowserRouter, Routes, Route, NavLink, Navigate, useNavigate } from 'react-router-dom';
import './App.css';
import Dashboard from './pages/Dashboard';
import Monthly from './pages/Monthly';
import Transactions from './pages/Transactions';
import Categories from './pages/Categories';
import { FaBars, FaQuestionCircle, FaCog } from 'react-icons/fa';

// ✨ 1. isDirty 상태를 전역적으로 관리하기 위한 Context 생성
const DirtyContext = createContext<{ isDirty: boolean; setIsDirty: (dirty: boolean) => void; } | null>(null);
export const useDirty = () => useContext(DirtyContext);

// ✨ 2. isDirty 상태를 확인하고 내비게이션을 처리하는 NavLink 래퍼 컴포넌트
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
    <div className="app-container">
      <div className={`sidebar-container ${isSidebarOpen ? 'open' : ''}`} onMouseLeave={closeSidebar}>
        <div className={`sidebar-wrapper ${isSidebarOpen ? 'open' : ''}`}>
          <aside className="sidebar">
            <div className="sidebar-header"><h2>MyData</h2></div>
            <nav className="sidebar-nav">
              <ul>
                <li><GuardedNavLink to="/dashboard">Dashboard</GuardedNavLink></li>
                <li><GuardedNavLink to="/monthly">Monthly</GuardedNavLink></li>
                <li><GuardedNavLink to="/transactions">Transactions</GuardedNavLink></li>
                <li><GuardedNavLink to="/categories">Categories</GuardedNavLink></li>
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
        </Routes>
      </main>
    </div>
  );
};

const App = () => {
  const [isDirty, setIsDirty] = useState(false);
  return (
    <BrowserRouter>
      <DirtyContext.Provider value={{ isDirty, setIsDirty }}>
        <AppContent />
      </DirtyContext.Provider>
    </BrowserRouter>
  );
};

export default App;