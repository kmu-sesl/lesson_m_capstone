# Updated by: jichoi, 2026.03.10
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 로드 및 전처리
mnist = tf.keras.datasets.mnist

# 훈련 데이터셋과 테스트 데이터셋 분리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 레이블 데이터를 원-핫 인코딩(One-hot encoding) 형태로 변환
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# 8-bit 이미지 픽셀값(0~255)을 0~1 사이의 실수 값으로 정규화(Normalization)
x_train, x_test = x_train/255.0, x_test/255.0
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 2. 합성곱 신경망(CNN) 모델 구성
model = tf.keras.models.Sequential([
    # 입력 데이터를 채널(Channel) 차원이 포함된 3D 텐서(28x28x1)로 변환
    tf.keras.layers.Reshape((28,28,1), input_shape=(28,28)),

    # 첫 번째 합성곱 계층: 3x3 필터 32개, Zero-padding 적용, ReLU 활성화 함수
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    # 두 번째 합성곱 계층: 3x3 필터 64개
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    # 세 번째 합성곱 계층: 3x3 필터 128개
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    # 다차원 특징 맵을 1차원 벡터로 평탄화
    tf.keras.layers.Flatten(),

    # 완전연결계층(Fully Connected Layer) 및 출력층
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

# 3. 모델 컴파일 및 학습
# 최적화: Adam, 손실함수: Categorical Crossentropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 모델 학습 진행
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=2)

# 모델 성능 평가
model.evaluate(x_test, y_test, verbose=2)

# 4. 모델 저장 (소스 파일의 폴더 위치 기준)
# 현재 실행 중인 스크립트 파일의 디렉토리 경로 추출
current_dir = os.path.dirname(os.path.abspath(__file__))
# 해당 디렉토리 경로와 파일명을 병합하여 최종 저장 경로 생성
save_path = os.path.join(current_dir, 'mnist_cnn.h5')

# 모델 저장
model.save(save_path)
print(f"\n[시스템 메시지] 모델이 소스 파일과 동일한 위치에 저장되었습니다: {save_path}")

# 5. 추론 및 시각화
# 테스트 데이터셋 중 무작위로 이미지 1장 선택
random_image = x_test[np.random.choice(len(x_test))]

# Keras 모델 입력 규격(Batch size 차원 포함)을 맞추기 위해 차원 확장 (28, 28) -> (1, 28, 28)
test_image = np.expand_dims(random_image, axis = 0)

# 선택된 이미지에 대한 클래스 확률 예측
predict = model.predict(test_image)

# 요청 사항 반영: 최종 콘솔 출력에 검출된 숫자 명시
print('\n[예측 결과]')
print('각 클래스별 확률 분포 (Softmax 출력): \n', predict)
print('▶ 검출된 숫자: ', np.argmax(predict, axis=1)[0])

# 이미지 시각화
plt.imshow(test_image[0], cmap='Greys')
plt.show()