# WSL2 환경 TensorFlow GPU 딥러닝 세팅 가이드 (Python 3.12)

다학제캡스톤 CNN 실습을 Local Windows PC에서 구동하기 위한 설정 가이드입니다. 
WSL2 환경에서 Python 3.12 가상 환경을 구축하고 TensorFlow가 GPU를 인식하도록 설정하는 전체 과정입니다.

## 1. 필수 패키지 설치
WSL2 터미널에 접속한 후, Python 3.12 및 가상 환경 구축에 필요한 패키지들을 설치합니다.

```bash
sudo apt install python3.12 python3.12-pip python3.12-venv

```

## 2. 가상 환경 생성 및 활성화

작업할 프로젝트 폴더로 이동하여 가상 환경(`venv`)을 생성하고 활성화합니다.

```bash
# 프로젝트 폴더로 이동 후 실행
python3.12 -m venv venv
source ./venv/bin/activate

```

## 3. TensorFlow 및 CUDA 패키지 설치

`pip`를 최신 버전으로 업데이트하고, NVIDIA CUDA 라이브러리가 포함된 TensorFlow를 설치합니다.

```bash
pip install --upgrade pip
pip install tensorflow[and-cuda]

```

## 4. 라이브러리 경로 설정 및 GPU 인식 확인

설치된 NVIDIA 라이브러리와 Windows 호스트의 WSL 라이브러리 경로를 환경 변수에 추가한 뒤, TensorFlow에서 GPU를 정상적으로 인식하는지 테스트합니다.

```bash
# 환경 변수 설정
export NV_LIBS=$(python3 -c "import os, glob; print(':'.join(glob.glob(os.path.join(os.environ['VIRTUAL_ENV'], 'lib', 'python*', 'site-packages', 'nvidia', '*', 'lib'))))")
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$NV_LIBS:$LD_LIBRARY_PATH

# GPU 인식 여부 확인
python -c "import tensorflow as tf; print('GPU 사용 가능 여부:', tf.config.list_physical_devices('GPU'))"

```

## 5. 가상 환경 활성화 시 자동 적용 (권장)

GPU가 정상적으로 인식된다면, 앞으로 가상 환경을 켤 때마다 환경 변수가 자동으로 세팅되도록 `activate` 스크립트를 수정합니다. 프로젝트 폴더 최상단에서 아래 명령어를 실행하세요.

```bash
cat << 'EOF' >> venv/bin/activate

# TensorFlow GPU 라이브러리 경로 자동 설정
export NV_LIBS=$(python3 -c "import os, glob; print(':'.join(glob.glob(os.path.join(os.environ['VIRTUAL_ENV'], 'lib', 'python*', 'site-packages', 'nvidia', '*', 'lib'))))")
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$NV_LIBS:$LD_LIBRARY_PATH
EOF

```

---

## 💡 참고 및 주의사항

* **Windows 호스트 드라이버:** WSL2 내부(Ubuntu 등)에 별도의 Linux용 그래픽 드라이버를 설치할 필요는 없습니다. **Windows 운영체제**에 최신 NVIDIA 드라이버가 깔려 있으면 4번 단계의 `/usr/lib/wsl/lib` 경로를 통해 자동으로 연결됩니다.
* **버전 호환성:** TensorFlow 2.16 이상부터 Python 3.12를 공식 지원하므로 문제없이 작동합니다.
* **환경 변수 백업 (선택):** 위 5번 단계의 스크립트는 기존 `LD_LIBRARY_PATH`를 덮어씁니다. 추후 환경 변수 충돌을 방지하기 위해 기존 경로를 백업해두고 싶다면, 5번 대신 아래 스크립트를 적용하는 것을 추천합니다.

```bash
cat << 'EOF' >> venv/bin/activate

# TensorFlow GPU 라이브러리 경로 자동 설정 및 백업
export OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export NV_LIBS=$(python3 -c "import os, glob; print(':'.join(glob.glob(os.path.join(os.environ['VIRTUAL_ENV'], 'lib', 'python*', 'site-packages', 'nvidia', '*', 'lib'))))")
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$NV_LIBS:$LD_LIBRARY_PATH
EOF

```



```

```