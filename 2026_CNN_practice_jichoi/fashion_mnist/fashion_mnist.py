# Updated by: jichoi, 2026.03.10
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

train_data = pd.read_csv('./fashion-mnist_train.csv')
test_data = pd.read_csv('./fashion-mnist_test.csv')
# print(train_data.head())

# 1.
np_train_data = np.array(train_data)
np_test_data = np.array(test_data)

# 2.
x_train,y_train = np_train_data[:,1:], np_train_data[:,:1]
x_test,y_test = np_test_data[:,1:], np_test_data[:,:1]

# plt.imshow(x_train[0].reshape(28,28),cmap='Greys')
# plt.show()

# 3.
x_train, x_test = x_train/255.0, x_test/255.0

# 4.
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((28,28,1), input_shape=(784,)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
print(model.summary())
#
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=2)
model.evaluate(x_test, y_test, verbose=2)

model.save('fashinon_mnist.h5')

# 무작위 이미지 선택 및 차원 확장
random_index = np.random.choice(len(x_test))
random_image = x_test[random_index]
test_image = np.expand_dims(random_image, axis=0)

# 실제 정답(Label) 확인 (y_test가 One-hot 인코딩 되어 있으므로 argmax 사용)
actual_label = np.argmax(y_test[random_index])

# 모델 예측
predict = model.predict(test_image)
predict_index = np.argmax(predict, axis=1)[0]
confidence = predict[0][predict_index] * 100  # 확신도(%)

# Fashion-MNIST 클래스 이름 정의 (순서 고정)
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# 콘솔 출력
print(f"\n[Prediction Result]")
print(f" - Predicted Class: {class_names[predict_index]} ({predict_index})")
print(f" - Actual Class:    {class_names[actual_label]} ({actual_label})")
print(f" - Confidence:      {confidence:.2f}%")

# 시각화 (타이틀에 결과 표시)
plt.figure(figsize=(5, 5))
plt.imshow(test_image[0].reshape(28, 28), cmap='Greys')
plt.title(f"Pred: {class_names[predict_index]} | Actual: {class_names[actual_label]}")
plt.axis('off')
plt.show()

