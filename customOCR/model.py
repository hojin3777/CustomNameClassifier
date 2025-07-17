from transformers import LayoutLMv3ForTokenClassification
from config import PRETRAINED_MODEL_NAME, NUM_LABELS, id2label, label2id

def get_model():
    """
    Hugging Face Hub에서 사전 학습된 LayoutLMv3 모델을 로드하고,
    우리 데이터셋의 라벨 수에 맞게 마지막 레이어를 수정한 모델을 반환합니다.
    """
    # from_pretrained를 사용하여 사전 학습된 가중치를 불러옵니다.
    # num_labels: 모델의 출력층(classification head)을 우리 라벨 수에 맞게 초기화합니다.
    # id2label, label2id: 모델 설정에 라벨 정보를 추가하여 나중에 예측 결과를 해석하기 용이하게 합니다.
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id
    )
    
    return model

# --- 이 파일이 직접 실행될 때 테스트용으로 동작하는 코드 ---
if __name__ == '__main__':
    print(f"'{PRETRAINED_MODEL_NAME}' 모델을 로딩합니다...")
    
    try:
        model = get_model()
        print("모델 로딩 완료!")
        
        # 모델의 구조 일부를 출력하여 확인합니다.
        # print("\n--- 모델 구조 (일부) ---")
        # print(model)
        
        # 특히 마지막 분류 레이어(classifier)의 출력 뉴런 수가 우리 라벨 수와 맞는지 확인합니다.
        print(f"\n분류 레이어(classifier): {model.classifier}")
        print(f"분류 레이어의 출력 뉴런 수: {model.classifier.out_features}")
        print(f"설정된 라벨 수 (from config.py): {NUM_LABELS}")
        
        if model.classifier.out_features == NUM_LABELS:
            print("\n[성공] 모델의 최종 출력 뉴런 수가 라벨 수와 일치합니다.")
        else:
            print("\n[실패] 모델의 최종 출력 뉴런 수가 라벨 수와 일치하지 않습니다. config.py를 확인해주세요.")

    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        print("인터넷 연결을 확인하고, config.py의 PRETRAINED_MODEL_NAME이 올바른지 확인해주세요.")
