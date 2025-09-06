import React from 'react';

const Dashboard = () => {
  return (
    <>
      <header className="main-header">
        <h1>Dashboard</h1>
      </header>
      <div className="content-area dashboard-grid">
        <div className="card chart-placeholder">
          <h3>월별 지출 추이</h3>
          <p>(차트 영역)</p>
        </div>
        <div className="card chart-placeholder">
          <h3>월별 입출금 흐름</h3>
          <p>(차트 영역)</p>
        </div>
      </div>
    </>
  );
};

export default Dashboard;