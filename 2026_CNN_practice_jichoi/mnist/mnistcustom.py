# Updated by: jichoi, 2026.03.10
# MNIST 모델은 애초에 '검정 바탕(0)에 흰색/회색(1~255)으로 쓰인 28x28 픽셀의 숫자'만 인식하도록 학습된 모델
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 1. 현재 소스 파일(.py)의 디렉토리 절대 경로 추출
current_dir = os.path.dirname(os.path.abspath(__file__))

# 추출된 디렉토리 경로를 기준으로 입력 이미지 및 모델 파일의 최종 경로 구성l
image_path = os.path.join(current_dir, 'number4.png')
# image_path = os.path.join(current_dir, 'number_160x160.png')
# image_path = os.path.join(current_dir, 'number_red_160x160.png')
model_path = os.path.join(current_dir, 'mnist_cnn.h5')

# 2. 이미지 파일 로딩 및 예외 처리 (한글 경로 호환성 확보)
# 파일 시스템에서 바이너리(8-bit unsigned integer) 스트림 형태로 데이터를 읽어옴
image_array = np.fromfile(image_path, np.uint8)

# 배열의 크기가 0인 경우 (파일이 없거나 읽기 권한이 없는 경우) 예외 처리
if image_array.size == 0:
    print(f"[오류] 파일을 찾을 수 없거나 읽을 수 없습니다.\n확인할 경로: {image_path}")
    sys.exit(1)

# 바이너리 배열 데이터를 OpenCV의 흑백(Grayscale) 이미지 행렬로 디코딩
image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

# 디코딩 실패 시 (파일이 손상되었거나 이미지 포맷이 아닌 경우) 예외 처리
if image is None:
    print(f"[오류] 이미지 파일을 디코딩할 수 없습니다. 파일 손상 여부를 확인하세요.")
    sys.exit(1)

# 3. 이미지 리사이징 (해상도 검사 및 변환)
# 입력 영상의 크기가 28x28이 아닐 경우 강제 리사이징 및 로그 출력
if image.shape != (28, 28):
    original_shape = image.shape
    # 영상 축소에 유리한 공간 보간법(INTER_AREA)을 사용하여 28x28 해상도로 변환
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    print(f"[시스템 로그] 입력 이미지의 해상도가 {original_shape}에서 (28, 28)로 리사이징 되었습니다.")

# 8-bit 영상 신호(0~255)를 0과 1 사이의 실수 범위로 선형 정규화(Normalization)
image = image / 255.0

# 4. 딥러닝 모델 로딩 및 예외 처리
try:
    # 저장된 Keras 모델 아키텍처 및 파라미터 복원
    model = tf.keras.models.load_model(model_path)
except OSError:
    # 경로에 모델 파일이 없거나 HDF5 포맷이 손상된 경우의 예외 처리
    print(f"[오류] 모델 파일을 찾을 수 없거나 로드할 수 없습니다.\n확인할 경로: {model_path}")
    sys.exit(1)

# 5. 차원 확장 및 추론 연산
# Keras 모델 입력 규격(배치 사이즈)에 맞추기 위해 차원 확장 (28, 28) -> (1, 28, 28)
test_image = np.expand_dims(image, axis=0)

# 피드포워드(Feed-forward) 연산을 통한 클래스별 확률 분포 추론
predict = model.predict(test_image)

# 6. 결과 출력 및 시각화
print('\n[예측 결과]')
print('각 클래스별 확률 분포 (Softmax 출력): \n', predict)
print('▶ 검출된 숫자: ', np.argmax(predict, axis=1)[0])

# matplotlib를 이용한 추론용 입력 데이터 시각화
plt.imshow(test_image[0], cmap='Greys')
plt.show()