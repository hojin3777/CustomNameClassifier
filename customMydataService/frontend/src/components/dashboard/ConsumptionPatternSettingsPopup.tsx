import React, { useState, useEffect } from 'react';
import './ConsumptionPatternSettingsPopup.css';

const API_BASE_URL = 'http://localhost:5000';

interface Settings {
  weekend_ratio_threshold: number;
  weekday_min_count: number;
  payday_spike_threshold: number;
  month_period_threshold: number;
  impulse_amount_limit: number;
  impulse_increase_threshold: number;
  category_spike_threshold: number;
  budget_alert_margin: number;
  no_spend_min_days: number;
  year_comparison_threshold: number;
  fixed_ratio_warning: number;
}

interface ConsumptionPatternSettingsPopupProps {
  onClose: () => void;
  onSave: () => void;
}

const ConsumptionPatternSettingsPopup: React.FC<ConsumptionPatternSettingsPopupProps> = ({ onClose, onSave }) => {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // fetchSettings: 설정 불러오기
  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/settings/consumption-pattern`);
        const data = await response.json();
        setSettings(data);
      } catch (error) {
        console.error('Failed to fetch settings:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSettings();
  }, []);

  // handleSave: 설정 저장
  const handleSave = async () => {
    if (!settings) return;

    try {
      await fetch(`${API_BASE_URL}/api/settings/consumption-pattern`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
      onSave();
      onClose();
    } catch (error) {
      console.error('Failed to save settings:', error);
      alert('설정 저장에 실패했습니다.');
    }
  };

  // handleChange: 값 변경
  const handleChange = (key: keyof Settings, value: number) => {
    if (!settings) return;
    setSettings({ ...settings, [key]: value });
  };

  // handleReset: 기본값으로 초기화
  const handleReset = () => {
    if (!window.confirm('기본값으로 초기화하시겠습니까?')) return;
    
    const defaultSettings: Settings = {
      weekend_ratio_threshold: 1.5,
      weekday_min_count: 3,
      payday_spike_threshold: 30,
      month_period_threshold: 40,
      impulse_amount_limit: 10000,
      impulse_increase_threshold: 50,
      category_spike_threshold: 100,
      budget_alert_margin: 10,
      no_spend_min_days: 3,
      year_comparison_threshold: 20,
      fixed_ratio_warning: 60
    };
    setSettings(defaultSettings);
  };

  if (isLoading) {
    return (
      <div className="popup-overlay-cpsettings">
        <div className="popup-container-cpsettings">
          <div className="popup-loading-cpsettings">설정 불러오는 중...</div>
        </div>
      </div>
    );
  }

  if (!settings) return null;

  return (
    <div className="popup-overlay-cpsettings" onClick={onClose}>
      <div className="popup-container-cpsettings" onClick={(e) => e.stopPropagation()}>
        <div className="popup-header-cpsettings">
          <h3>소비 패턴 인사이트 설정</h3>
          <button className="popup-close-button-cpsettings" onClick={onClose}>×</button>
        </div>

        <div className="popup-body-cpsettings">
          <div className="settings-group-cpsettings">
            <h4>주말/평일 소비 비교</h4>
            <div className="setting-item-cpsettings">
              <label>비율 임계값 (배)</label>
              <input
                type="number"
                step="0.1"
                value={settings.weekend_ratio_threshold}
                onChange={(e) => handleChange('weekend_ratio_threshold', parseFloat(e.target.value))}
              />
            </div>
          </div>

          <div className="settings-group-cpsettings">
            <h4>요일별 집중 소비</h4>
            <div className="setting-item-cpsettings">
              <label>최소 거래 횟수 (회)</label>
              <input
                type="number"
                value={settings.weekday_min_count}
                onChange={(e) => handleChange('weekday_min_count', parseInt(e.target.value))}
              />
            </div>
          </div>

          <div className="settings-group-cpsettings">
            <h4>급여일 후 지출 증가</h4>
            <div className="setting-item-cpsettings">
              <label>증가율 임계값 (%)</label>
              <input
                type="number"
                value={settings.payday_spike_threshold}
                onChange={(e) => handleChange('payday_spike_threshold', parseInt(e.target.value))}
              />
            </div>
          </div>

          <div className="settings-group-cpsettings">
            <h4>월초/월말 소비 차이</h4>
            <div className="setting-item-cpsettings">
              <label>차이 임계값 (%)</label>
              <input
                type="number"
                value={settings.month_period_threshold}
                onChange={(e) => handleChange('month_period_threshold', parseInt(e.target.value))}
              />
            </div>
          </div>

          <div className="settings-group-cpsettings">
            <h4>소액 다빈도 지출</h4>
            <div className="setting-item-cpsettings">
              <label>소액 기준 (원)</label>
              <input
                type="number"
                step="1000"
                value={settings.impulse_amount_limit}
                onChange={(e) => handleChange('impulse_amount_limit', parseInt(e.target.value))}
              />
            </div>
            <div className="setting-item-cpsettings">
              <label>증가율 임계값 (%)</label>
              <input
                type="number"
                value={settings.impulse_increase_threshold}
                onChange={(e) => handleChange('impulse_increase_threshold', parseInt(e.target.value))}
              />
            </div>
          </div>

          <div className="settings-group-cpsettings">
            <h4>카테고리 급증</h4>
            <div className="setting-item-cpsettings">
              <label>증가율 임계값 (%)</label>
              <input
                type="number"
                value={settings.category_spike_threshold}
                onChange={(e) => handleChange('category_spike_threshold', parseInt(e.target.value))}
              />
            </div>
          </div>

          <div className="settings-group-cpsettings">
            <h4>예산 초과 경고</h4>
            <div className="setting-item-cpsettings">
              <label>초과 여유분 (%)</label>
              <input
                type="number"
                value={settings.budget_alert_margin}
                onChange={(e) => handleChange('budget_alert_margin', parseInt(e.target.value))}
              />
            </div>
          </div>

          <div className="settings-group-cpsettings">
            <h4>무지출 챌린지</h4>
            <div className="setting-item-cpsettings">
              <label>최소 연속 일수 (일)</label>
              <input
                type="number"
                value={settings.no_spend_min_days}
                onChange={(e) => handleChange('no_spend_min_days', parseInt(e.target.value))}
              />
            </div>
          </div>

          <div className="settings-group-cpsettings">
            <h4>전년 대비 변화</h4>
            <div className="setting-item-cpsettings">
              <label>변화율 임계값 (%)</label>
              <input
                type="number"
                value={settings.year_comparison_threshold}
                onChange={(e) => handleChange('year_comparison_threshold', parseInt(e.target.value))}
              />
            </div>
          </div>

          <div className="settings-group-cpsettings">
            <h4>고정비 비중 경고</h4>
            <div className="setting-item-cpsettings">
              <label>비중 임계값 (%)</label>
              <input
                type="number"
                value={settings.fixed_ratio_warning}
                onChange={(e) => handleChange('fixed_ratio_warning', parseInt(e.target.value))}
              />
            </div>
          </div>
        </div>

        <div className="popup-footer-cpsettings">
          <button className="reset-button-cpsettings" onClick={handleReset}>기본값 복원</button>
          <div className="action-buttons-cpsettings">
            <button className="cancel-button-cpsettings" onClick={onClose}>취소</button>
            <button className="save-button-cpsettings" onClick={handleSave}>저장</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConsumptionPatternSettingsPopup;