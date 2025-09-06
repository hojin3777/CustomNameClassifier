import React from 'react';

const Monthly = () => {
  const days = ['일', '월', '화', '수', '목', '금', '토'];
  const calendarCells = Array.from({ length: 35 }); // 5주 * 7일

  return (
    <>
      <header className="main-header">
        <div className="month-selector">
          {/* TODO: Dropdown 기능 구현 */}
          <select defaultValue="2025">
            <option>2025년</option>
            <option>2024년</option>
          </select>
          <select defaultValue="9">
            <option>9월</option>
            <option>8월</option>
          </select>
        </div>
      </header>
      <div className="content-area calendar-grid">
        {days.map(day => <div key={day} className="calendar-header">{day}</div>)}
        {calendarCells.map((_, index) => <div key={index} className="calendar-cell"></div>)}
      </div>
    </>
  );
};

export default Monthly;