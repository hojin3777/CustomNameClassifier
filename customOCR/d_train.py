import os
import subprocess

# --- 설정 ---
# Donut 프로젝트 마스터 폴더 경로
DONUT_MASTER_DIR = 'C:/code/donut-master/'
# 방금 생성한 학습 설정 파일 경로
CONFIG_FILE_PATH = 'C:/code/customOCR/d_train_config.yaml'
# Donut 전용으로 만든 가상환경의 Python 실행 파일 경로
# (가상환경을 사용하지 않는 경우, 'python'으로 두어도 무방)
PYTHON_EXECUTABLE = 'C:/code/.venv_donut/Scripts/python.exe'

def train_donut_model():
    """
    설정 파일을 사용하여 Donut 모델 학습을 시작합니다.
    """
    print("Donut 모델 학습을 시작합니다...")
    print(f"  - Donut 프로젝트 경로: {DONUT_MASTER_DIR}")
    print(f"  - 설정 파일: {CONFIG_FILE_PATH}")

    # donut-master 폴더의 train.py를 실행하는 명령어 생성
    # 형식: python train.py --config {설정파일경로}
    command = [
        PYTHON_EXECUTABLE,
        os.path.join(DONUT_MASTER_DIR, 'train.py'),
        '--config',
        CONFIG_FILE_PATH
    ]

    try:
        # 서브프로세스로 학습 스크립트 실행
        # cwd: train.py가 있는 donut-master 폴더에서 명령어를 실행하도록 설정
        process = subprocess.Popen(command, cwd=DONUT_MASTER_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')

        # 학습 과정의 출력을 실시간으로 터미널에 표시
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        if rc == 0:
            print("\n🎉 Donut 모델 학습이 성공적으로 완료되었습니다.")
        else:
            print(f"\n오류: Donut 모델 학습 중 문제가 발생했습니다. (Exit code: {rc})")

    except FileNotFoundError:
        print(f"오류: '{PYTHON_EXECUTABLE}' 또는 'train.py'를 찾을 수 없습니다. 경로를 확인해주세요.")
    except Exception as e:
        print(f"학습 실행 중 예외가 발생했습니다: {e}")

if __name__ == '__main__':
    train_donut_model()