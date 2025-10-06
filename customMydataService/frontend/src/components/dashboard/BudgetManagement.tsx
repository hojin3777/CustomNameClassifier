import React, { useState, useEffect, useCallback, useRef } from 'react';
import { FaPlusCircle, FaPen, FaTrash } from 'react-icons/fa';
import ConfirmPopup from '../ConfirmPopup';
import FloatingSelectPopup, { type FloatingSelectHandle, type Opt } from '../FloatingSelectPopup';
import './BudgetManagement.css';

const API_BASE_URL = 'http://localhost:5000';

// ******************** 타입 정의 ********************
interface MajorCategory { id: number; name: string; }
interface MinorCategory { uuid: string; name: string; major_category_id: number; }
interface Budget {
    id: number;
    budget_type: 'major' | 'minor';
    target_id: string;
    amount: number;
    target_name: string;
    major_category_id: number;
    major_category_name: string;
}
interface BudgetWithSpending extends Budget {
    spent_amount: number;
}
interface BudgetDraft {
    tempId?: string;
    isNew?: boolean;
    major_category_id?: number;
    minor_category_uuid?: string;
    amount?: number;
}
interface BudgetManagementProps {
    selectedYear: number | null;
    selectedMonth: number | null;
}
interface TreemapMajor {
    name: string;
    value: number;
    children: TreemapMinor[];
}
interface TreemapMinor {
    name: string;
    value: number;
}

// ******************** 메인 컴포넌트 ********************
const BudgetManagement: React.FC<BudgetManagementProps> = ({ selectedYear, selectedMonth }) => {
    const [budgets, setBudgets] = useState<BudgetWithSpending[]>([]);
    const [drafts, setDrafts] = useState<BudgetDraft[]>([]);
    const [majorCategories, setMajorCategories] = useState<MajorCategory[]>([]);
    const [minorCategories, setMinorCategories] = useState<MinorCategory[]>([]);
    const [editingCell, setEditingCell] = useState<{ id: string | number, field: string } | null>(null);
    const [editValue, setEditValue] = useState<string>('');
    const [isConfirmOpen, setIsConfirmOpen] = useState(false);
    const [confirmAction, setConfirmAction] = useState<{ onConfirm: () => void } | null>(null);
    const floatingSelectRef = useRef<FloatingSelectHandle>(null);

    // ******************** 데이터 로딩 ********************
    // fetchCategories: 카테고리 목록 로드 (고정수입/유동수입/이체분류 제외)
    const fetchCategories = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/api/categories`);
            const data = await res.json();
            const allCategories: MajorCategory[] = data.map((cat: any) => ({ id: cat.id, name: cat.name }));
            const filteredMajor = allCategories.filter((mc: MajorCategory) =>
                !['고정수입', '유동수입', '이체분류'].includes(mc.name)
            );
            setMajorCategories(filteredMajor);

            const allMinorCategories: MinorCategory[] = data.flatMap((cat: any) =>
                cat.minors.map((m: any) => ({ ...m, major_category_id: cat.id }))
            );
            setMinorCategories(allMinorCategories);
        } catch (error) {
            console.error('카테고리 로딩 실패:', error);
        }
    }, []);

    // fetchData: Treemap 데이터 + 예산 데이터를 가져와 조합
    const fetchData = useCallback(async () => {
        if (!selectedYear || !selectedMonth) return;

        try {
            // 1. Treemap 데이터 가져오기 (지출 집계)
            const treemapRes = await fetch(
                `${API_BASE_URL}/api/statistics/category_treemap?year=${selectedYear}&month=${selectedMonth}`
            );
            const treemapResult = await treemapRes.json();
            const treemapData: TreemapMajor[] = treemapResult.data || [];

            // 2. 예산 설정 가져오기 (별도 API)
            const budgetRes = await fetch(`${API_BASE_URL}/api/budgets`);
            const budgetData: Budget[] = await budgetRes.json();

            // 3. 두 데이터를 조합
            const combinedBudgets: BudgetWithSpending[] = budgetData.map((budget) => {
                let spent_amount = 0;

                if (budget.budget_type === 'major') {
                    // Treemap에서 해당 대분류 찾기
                    const majorData = treemapData.find((m) => m.name === budget.target_name);
                    spent_amount = majorData?.value || 0;
                } else {
                    // Treemap에서 해당 소분류 찾기
                    for (const major of treemapData) {
                        const minorData = major.children?.find((c) => c.name === budget.target_name);
                        if (minorData) {
                            spent_amount = minorData.value;
                            break;
                        }
                    }
                }

                return {
                    ...budget,
                    spent_amount
                };
            });

            setBudgets(combinedBudgets);
        } catch (error) {
            console.error('데이터 로딩 실패:', error);
        }
    }, [selectedYear, selectedMonth]);

    useEffect(() => {
        fetchCategories();
    }, [fetchCategories]);

    useEffect(() => {
        if (selectedYear && selectedMonth) {
            fetchData();
        }
    }, [selectedYear, selectedMonth, fetchData]);

    // ******************** 핸들러: 행 추가 ********************
    const handleAddRow = () => {
        if (budgets.length + drafts.length >= 30) {
            alert('예산은 최대 30개까지 설정할 수 있습니다.');
            return;
        }
        const newDraft: BudgetDraft = {
            tempId: `temp-${Date.now()}`,
            isNew: true,
            major_category_id: undefined,
            minor_category_uuid: undefined,
            amount: 0
        };
        setDrafts((prev) => [...prev, newDraft]);
    };

    // ******************** 핸들러: 카테고리 선택 ********************
    const handleCategoryChange = async (
        item: BudgetWithSpending | BudgetDraft,
        field: 'major' | 'minor',
        value: string | number,
        isDb: boolean
    ) => {
        if (isDb) {
            // DB 예산 수정
            const budget = item as BudgetWithSpending;
            const currentMajorId = budget.major_category_id;
            const currentMinorUuid = budget.budget_type === 'minor' ? budget.target_id : undefined;

            let newMajorId = currentMajorId;
            let newMinorUuid = currentMinorUuid;

            if (field === 'major') {
                newMajorId = Number(value);
                newMinorUuid = undefined; // 대분류 변경 시 소분류 초기화
            } else {
                newMinorUuid = value ? String(value) : undefined;
            }

            // 백엔드 저장
            const budget_type = newMinorUuid ? 'minor' : 'major';
            const target_id = newMinorUuid || String(newMajorId);

            try {
                await fetch(`${API_BASE_URL}/api/budgets/${budget.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ budget_type, target_id, amount: budget.amount })
                });
                await fetchData(); // 데이터 재로드
            } catch (error) {
                console.error('예산 카테고리 수정 실패:', error);
                alert('예산 카테고리 수정에 실패했습니다.');
            }
        } else {
            // Draft 수정
            const draft = item as BudgetDraft;
            const updatedDraft = { ...draft };

            if (field === 'major') {
                updatedDraft.major_category_id = Number(value);
                updatedDraft.minor_category_uuid = undefined; // 대분류 변경 시 소분류 초기화
            } else {
                updatedDraft.minor_category_uuid = value ? String(value) : undefined;
            }

            setDrafts((prev) =>
                prev.map((d) => (d.tempId === draft.tempId ? updatedDraft : d))
            );

            // 대분류 + 금액이 있으면 자동 저장
            if (updatedDraft.major_category_id && updatedDraft.amount && updatedDraft.amount > 0) {
                await saveDraft(updatedDraft);
            }
        }
    };

    // ******************** 핸들러: 금액 편집 ********************
    const handleEditStart = (budget: BudgetWithSpending | BudgetDraft, isDb: boolean) => {
        const id = isDb ? (budget as BudgetWithSpending).id : (budget as BudgetDraft).tempId!;
        setEditingCell({ id, field: 'amount' });
        setEditValue(String((budget as any).amount || ''));
    };

    const handleEditSave = async () => {
        if (!editingCell) return;

        const { id } = editingCell;
        const isDb = typeof id === 'number';

        if (isDb) {
            // DB에 있는 예산 수정
            const budget = budgets.find((b) => b.id === id);
            if (!budget) return;

            const newAmount = parseInt(editValue, 10) || 0;
            if (newAmount <= 0) {
                alert('0보다 큰 예산 금액을 입력해주세요.');
                setEditingCell(null);
                return;
            }

            try {
                await fetch(`${API_BASE_URL}/api/budgets/${budget.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        budget_type: budget.budget_type,
                        target_id: budget.target_id,
                        amount: newAmount
                    })
                });
                await fetchData();
            } catch (error) {
                console.error('예산 수정 실패:', error);
                alert('예산 수정에 실패했습니다.');
            }
        } else {
            // 임시 Draft 수정
            const draft = drafts.find((d) => d.tempId === id);
            if (!draft) return;

            const newAmount = parseInt(editValue, 10) || 0;
            draft.amount = newAmount;

            setDrafts((prev) => prev.map((d) => (d.tempId === id ? draft : d)));

            // 대분류 + 금액이 있으면 자동 저장
            if (draft.major_category_id && newAmount > 0) {
                await saveDraft(draft);
            }
        }

        setEditingCell(null);
    };

    // ******************** 핸들러: Draft 저장 ********************
    const saveDraft = async (draft: BudgetDraft) => {
        if (!draft.major_category_id || !draft.amount || draft.amount <= 0) {
            return; // 필수 값이 없으면 저장 안 함
        }

        const budget_type = draft.minor_category_uuid ? 'minor' : 'major';
        const target_id = draft.minor_category_uuid || String(draft.major_category_id);

        try {
            await fetch(`${API_BASE_URL}/api/budgets`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ budget_type, target_id, amount: draft.amount })
            });

            // Draft 제거 및 데이터 재로드
            setDrafts((prev) => prev.filter((d) => d.tempId !== draft.tempId));
            await fetchData();
        } catch (error) {
            console.error('예산 저장 실패:', error);
            alert('예산 저장에 실패했습니다.');
        }
    };

    // ******************** 핸들러: 삭제 ********************
    const handleDelete = (budget: BudgetWithSpending | BudgetDraft, isDb: boolean) => {
        if (!isDb) {
            // Draft 삭제
            setDrafts((prev) => prev.filter((d) => d.tempId !== (budget as BudgetDraft).tempId));
            return;
        }

        // DB 예산 삭제 (확인 팝업)
        setConfirmAction({
            onConfirm: async () => {
                try {
                    await fetch(`${API_BASE_URL}/api/budgets/${(budget as BudgetWithSpending).id}`, {
                        method: 'DELETE'
                    });
                    await fetchData();
                } catch (error) {
                    console.error('예산 삭제 실패:', error);
                } finally {
                    setIsConfirmOpen(false);
                    setConfirmAction(null);
                }
            }
        });
        setIsConfirmOpen(true);
    };

    // ******************** 핸들러: FloatingSelectPopup ********************
    const openCategoryPopup = (
        e: React.MouseEvent,
        item: BudgetWithSpending | BudgetDraft,
        type: 'major' | 'minor',
        isDb: boolean
    ) => {
        const cell = e.currentTarget as HTMLElement;
        const rect = cell.getBoundingClientRect();
        const position = { top: rect.bottom, left: rect.left, width: rect.width };

        if (type === 'major') {
            const options = majorCategories.map((c) => ({ value: String(c.id), label: c.name }));
            const currentValue = isDb
                ? String((item as BudgetWithSpending).major_category_id)
                : (item as BudgetDraft).major_category_id
                    ? String((item as BudgetDraft).major_category_id)
                    : '';

            floatingSelectRef.current?.open(
                options,
                currentValue,
                position,
                (value: string) => handleCategoryChange(item, 'major', Number(value), isDb),
                '-- 대분류 --'
            );
        } else {
            const majorId = isDb
                ? (item as BudgetWithSpending).major_category_id
                : (item as BudgetDraft).major_category_id;

            if (!majorId) return;

            const minorOptions = getMinorOptions(majorId);
            const options = [
                { value: '', label: '(대분류 전체)' },
                ...minorOptions.map((c) => ({ value: c.uuid, label: c.name }))
            ];

            const currentValue = isDb
                ? (item as BudgetWithSpending).budget_type === 'minor'
                    ? (item as BudgetWithSpending).target_id
                    : ''
                : (item as BudgetDraft).minor_category_uuid || '';

            floatingSelectRef.current?.open(
                options,
                currentValue,
                position,
                (value: string) => handleCategoryChange(item, 'minor', value, isDb),
                '(대분류 전체)'
            );
        }
    };

    // ******************** 유틸리티 함수 ********************
    const getProgressBarColor = (percentage: number, index: number) => {
        if (percentage > 100) return 'var(--color-highlight-2)'; // 빨간색
        const colors = [
            'var(--color-highlight-3)', // 주황
            'var(--color-highlight-4)', // 초록
            'var(--color-highlight-5)', // 파랑
            'var(--color-highlight-6)', // 핑크
            'var(--color-highlight-1)'  // 회색
        ];
        return colors[index % colors.length];
    };

    const getMinorOptions = (majorId: number) => {
        return minorCategories.filter((mc) => mc.major_category_id === majorId);
    };

    const getMajorCategoryName = (id: number | undefined) => {
        if (!id) return null;
        return majorCategories.find((mc) => mc.id === id)?.name || null;
    };

    const getMinorCategoryName = (draft: BudgetDraft) => {
        if (!draft.minor_category_uuid) return null;
        return minorCategories.find((mc) => mc.uuid === draft.minor_category_uuid)?.name || null;
    };

    // ******************** 렌더링 ********************
    return (
        <div className="dashboard-card">
            <div className="dashboard-card-header">
                <h3 className="dashboard-card-title">예산 관리</h3>
                <div className="dashboard-card-subtitle">
                    {selectedYear && selectedMonth
                        ? `${selectedYear}년 ${selectedMonth}월 기준`
                        : '연도와 월을 선택하세요'}
                </div>
            </div>
            <div className="dashboard-card-content budget-card-content-mbudget">
                <div className="budget-table-container-mbudget">
                    <table className="budget-table-mbudget">
                        <thead>
                            <tr>
                                <th className="col-major-mbudget">대분류</th>
                                <th className="col-minor-mbudget">소분류</th>
                                <th className="col-spent-mbudget">사용금액</th>
                                <th className="col-progress-mbudget">사용률</th>
                                <th className="col-amount-mbudget">예산 설정</th>
                            </tr>
                        </thead>
                        <tbody>
                            {/* DB 예산 렌더링 */}
                            {budgets.map((budget, index) => {
                                const isEditing = editingCell?.id === budget.id && editingCell?.field === 'amount';
                                const percentage = budget.amount > 0
                                    ? (budget.spent_amount / budget.amount) * 100
                                    : 0;

                                return (
                                    <tr key={budget.id}>
                                        <td
                                            className="clickable-cell-mbudget"
                                            onClick={(e) => openCategoryPopup(e, budget, 'major', true)} // 팝업 추가
                                        >
                                            {budget.major_category_name}
                                        </td>
                                        <td
                                            className="clickable-cell-mbudget"
                                            onClick={(e) => openCategoryPopup(e, budget, 'minor', true)} // 팝업 추가
                                        >
                                            {budget.budget_type === 'minor' ? (
                                                budget.target_name
                                            ) : (
                                                <span className="whole-category-mbudget">(대분류 전체)</span>
                                            )}
                                        </td>
                                        <td className="spent-cell-mbudget">
                                            {budget.spent_amount.toLocaleString()}
                                        </td>
                                        <td className="progress-cell-mbudget">
                                            <div className="percentage-cell-mbudget">
                                                <div
                                                    className="percentage-bar-fill-mbudget"
                                                    style={{
                                                        width: `${Math.min(percentage, 100)}%`,
                                                        backgroundColor: getProgressBarColor(percentage, index)
                                                    }}
                                                ></div>
                                                <span className="percentage-text-mbudget">
                                                    {percentage.toFixed(1)}%
                                                </span>
                                            </div>
                                        </td>
                                        <td className="amount-cell-mbudget">
                                            {isEditing ? (
                                                <input
                                                    type="text"
                                                    className="edit-input-mbudget"
                                                    value={editValue}
                                                    onChange={(e) =>
                                                        setEditValue(e.target.value.replace(/[^0-9]/g, ''))
                                                    }
                                                    onBlur={handleEditSave}
                                                    onKeyDown={(e) => {
                                                        if (e.key === 'Enter') handleEditSave();
                                                        if (e.key === 'Escape') setEditingCell(null);
                                                    }}
                                                    autoFocus
                                                />
                                            ) : (
                                                <>
                                                    <span className="amount-value-mbudget">
                                                        {budget.amount.toLocaleString()}
                                                    </span>
                                                    <div className="cell-actions-mbudget">
                                                        <FaPen
                                                            className="action-icon-mbudget edit-icon-mbudget"
                                                            onClick={() => handleEditStart(budget, true)}
                                                        />
                                                        <FaTrash
                                                            className="action-icon-mbudget delete-icon-mbudget"
                                                            onClick={() => handleDelete(budget, true)}
                                                        />
                                                    </div>
                                                </>
                                            )}
                                        </td>
                                    </tr>
                                );
                            })}

                            {/* Draft 렌더링 */}
                            {drafts.map((draft) => {
                                const isEditing = editingCell?.id === draft.tempId && editingCell?.field === 'amount';
                                const majorName = getMajorCategoryName(draft.major_category_id);
                                const minorName = getMinorCategoryName(draft);

                                return (
                                    <tr key={draft.tempId}>
                                        <td
                                            className="clickable-cell-mbudget"
                                            onClick={(e) => openCategoryPopup(e, draft, 'major', false)} // isDb: false
                                        >
                                            {majorName || <span className="placeholder-mbudget">-- 대분류 --</span>}
                                        </td>
                                        <td
                                            className="clickable-cell-mbudget"
                                            onClick={(e) => openCategoryPopup(e, draft, 'minor', false)} // isDb: false
                                        >
                                            {draft.major_category_id ? (
                                                minorName ? (
                                                    minorName
                                                ) : (
                                                    <span className="whole-category-mbudget">(대분류 전체)</span>
                                                )
                                            ) : (
                                                <span className="placeholder-mbudget">-- 소분류 --</span>
                                            )}
                                        </td>
                                        <td className="spent-cell-mbudget">-</td>
                                        <td className="progress-cell-mbudget">
                                            <div className="percentage-cell-mbudget">
                                                <span className="percentage-text-mbudget">-</span>
                                            </div>
                                        </td>
                                        <td className="amount-cell-mbudget">
                                            {isEditing ? (
                                                <input
                                                    type="text"
                                                    className="edit-input-mbudget"
                                                    value={editValue}
                                                    onChange={(e) =>
                                                        setEditValue(e.target.value.replace(/[^0-9]/g, ''))
                                                    }
                                                    onBlur={handleEditSave}
                                                    onKeyDown={(e) => {
                                                        if (e.key === 'Enter') handleEditSave();
                                                        if (e.key === 'Escape') setEditingCell(null);
                                                    }}
                                                    autoFocus
                                                />
                                            ) : (
                                                <>
                                                    <span className="amount-value-mbudget">
                                                        {(draft.amount || 0).toLocaleString()}
                                                    </span>
                                                    <div className="cell-actions-mbudget">
                                                        <FaPen
                                                            className="action-icon-mbudget edit-icon-mbudget"
                                                            onClick={() => handleEditStart(draft, false)}
                                                        />
                                                        <FaTrash
                                                            className="action-icon-mbudget delete-icon-mbudget"
                                                            onClick={() => handleDelete(draft, false)}
                                                        />
                                                    </div>
                                                </>
                                            )}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>

                    {/* 행 추가 버튼 */}
                    {budgets.length + drafts.length < 30 && (
                        <div className="add-row-button-mbudget" onClick={handleAddRow}>
                            <FaPlusCircle />
                            <span>예산 항목 추가</span>
                        </div>
                    )}
                </div>
            </div>

            {/* 팝업 */}
            <ConfirmPopup
                isOpen={isConfirmOpen}
                type="confirm"
                title="예산 삭제"
                message="해당 예산 항목을 삭제하시겠습니까?"
                onConfirm={() => confirmAction?.onConfirm()}
                onCancel={() => {
                    setIsConfirmOpen(false);
                    setConfirmAction(null);
                }}
            />
            <FloatingSelectPopup ref={floatingSelectRef} />
        </div>
    );
};

export default BudgetManagement;