""" cat/dog classification 2nd version """
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

import matplotlib.pyplot as plt

###########################################################
# 1) tensorflow 에서 GPU enable 되어 있는지 여부 확인
# 2) 현재 PC에서 가용 가능한 GPU 있는지 여부 확인
###########################################################
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
print("1) GPUs:", gpus)
print("2) check gpus:", tf.config.list_physical_devices('GPU'))

###########################################################
# (0) 전역 선택 변수 (옵션)
###########################################################
MODEL_TO_RUN = "baseline"     # 어떤 예제를 구동할지 선택, "baseline" | "aug" | "vgg"
USE_TFDATA = True             # (1) tf.data + prefetch 사용 (깨진 이미지 자동 스킵 포함)
USE_MIXED_PRECISION = True   # (2) mixed precision 사용, GPU 있을 경우에만 'True' 할것.
SAVE_PLOTS = True
PLOTS_DIR = "./plots"

###########################################################
# (2) Mixed Precision 옵션
###########################################################
if USE_MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed Precision: ON (mixed_float16)")
else:
    print("Mixed Precision: OFF")

###########################################################
# 공통 유틸
###########################################################
def plot_history(history, title_prefix="", save_prefix="result"):
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs_range = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    axes[0].plot(epochs_range, acc, label='Training Accuracy')
    axes[0].plot(epochs_range, val_acc, label='Validation Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].set_title('Accuracy')

    # Loss
    axes[1].plot(epochs_range, loss, label='Training Loss')
    axes[1].plot(epochs_range, val_loss, label='Validation Loss')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Loss')

    # ✅ 상단 제목: 모델명 + 날짜/시간
    fig.suptitle(f"{title_prefix}Training and Validation  |  {ts}", y=0.98)

    # ✅ 하단에도 작게 한번 더 찍고 싶으면(선택)
    fig.text(0.5, 0.01, ts, ha='center', va='bottom', fontsize=9)

    # ✅ 겹침 방지(상단/하단 공간 확보)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])

    if SAVE_PLOTS:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        # ✅ 파일명에도 타임스탬프 추가
        save_path = os.path.join(PLOTS_DIR, f"{save_prefix}_history_{ts_file}.png")
        fig.savefig(save_path, dpi=200)
        print("Saved plot:", save_path)

    plt.show()
    plt.close(fig)

###########################################################
# 1) 데이터 경로/기본 설정 (공통)
###########################################################
PATH = './kaggle_pet_images'
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)
print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

batch_size = 32
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

###########################################################
# (4) Data Augmentation을 모델 내부 레이어로 구성
###########################################################
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.2),
], name="data_augmentation")

###########################################################
# (1) tf.data 입력 파이프라인 (깨진 이미지 자동 스킵 포함)
###########################################################
def _safe_decode_with_ok(path_str, label_int, img_h, img_w):
    """
    성공: (H,W,3) float32 이미지, label(float32), ok(True)
    실패: 더미 이미지(0으로 채움), label, ok(False)
    """
    try:
        img_bytes = tf.io.read_file(path_str)

        # ❌ 기존 코드: 무조건 JPEG로만 디코딩하려고 시도함
        # img = tf.io.decode_jpeg(img_bytes, channels=3)
        # ✅ 수정된 코드: 이미지 형식을 자동으로 판별하여 디코딩 (BMP, PNG 등 모두 호환)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)

        img = tf.image.resize(img, [img_h, img_w])
        img = tf.cast(img, tf.float32) / 255.0
        y = tf.cast(label_int, tf.float32)
        ok = tf.constant(True)
        return img, y, ok
    except Exception:
        img = tf.zeros([img_h, img_w, 3], tf.float32)
        y = tf.cast(label_int, tf.float32)
        ok = tf.constant(False)
        return img, y, ok

def make_datasets_tfdata_safe(train_dir, validation_dir, img_size, batch_size):
    def list_files_with_labels(base_dir):
        cats_dir = os.path.join(base_dir, "cats")
        dogs_dir = os.path.join(base_dir, "dogs")

        cats = [os.path.join(cats_dir, f) for f in os.listdir(cats_dir)]
        dogs = [os.path.join(dogs_dir, f) for f in os.listdir(dogs_dir)]

        paths = cats + dogs
        labels = [0] * len(cats) + [1] * len(dogs)
        return paths, labels

    train_paths, train_labels = list_files_with_labels(train_dir)
    val_paths, val_labels = list_files_with_labels(validation_dir)

    print("tf.data train files:", len(train_paths))
    print("tf.data val files:", len(val_paths))

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels)).shuffle(len(train_paths))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

    H, W = img_size

    def map_fn(p, y):
        img, yy, ok = tf.py_function(
            func=lambda pp, yy: _safe_decode_with_ok(
                pp.numpy().decode('utf-8'),
                int(yy.numpy()),
                H, W
            ),
            inp=[p, y],
            Tout=(tf.float32, tf.float32, tf.bool)
        )
        # ✅ shape 고정(성공/실패 모두 동일 shape로 반환되므로 reshape 불필요)
        img.set_shape([H, W, 3])
        yy.set_shape([])
        ok.set_shape([])
        return img, yy, ok

    train_ds = train_ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # ✅ 실패 샘플은 ok=False로 완전히 제거
    train_ds = train_ds.filter(lambda x, y, ok: ok).map(lambda x, y, ok: (x, y),
                                                        num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.filter(lambda x, y, ok: ok).map(lambda x, y, ok: (x, y),
                                                    num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


###########################################################
# ImageDataGenerator (tf.data 안 쓰는 경우)
###########################################################
def make_generators_imagedatagenerator():
    from keras.preprocessing.image import ImageDataGenerator

    train_image_generator = ImageDataGenerator(rescale=1./255)
    validation_image_generator = ImageDataGenerator(rescale=1./255)

    train_gen = train_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=train_dir,
        shuffle=True,
        target_size=IMG_SIZE,
        class_mode='binary'
    )

    val_gen = validation_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=validation_dir,
        target_size=IMG_SIZE,
        class_mode='binary'
    )

    return train_gen, val_gen

###########################################################
# 공통 입력 소스 선택
###########################################################
if USE_TFDATA:
    train_input, val_input = make_datasets_tfdata_safe(
        train_dir=train_dir,
        validation_dir=validation_dir,
        img_size=IMG_SIZE,
        batch_size=batch_size
    )
else:
    train_input, val_input = make_generators_imagedatagenerator()

###########################################################
# 모델 생성 함수들
###########################################################
def _out_dtype():
    # mixed precision일 때 출력은 float32로 고정 권장
    return 'float32' if USE_MIXED_PRECISION else None

def build_baseline_model():
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid', dtype=_out_dtype())
    ])
    return model
def build_aug_model():
    model = Sequential([
        tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        data_aug,
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid', dtype=_out_dtype())
    ])
    return model

def build_vgg_model():
    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet',
                                      input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    vgg.trainable = False

    model = Sequential([
        tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        data_aug,
        vgg,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid', dtype=_out_dtype())
    ])
    return model, vgg

###########################################################
# 모델 선택 및 학습
###########################################################
if MODEL_TO_RUN == "baseline":
    model = build_baseline_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(
        train_input,
        epochs=epochs,
        validation_data=val_input,
        verbose=2
    )
    plot_history(history, title_prefix="Baseline. ", save_prefix="baseline")

elif MODEL_TO_RUN == "aug":
    model = build_aug_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(
        train_input,
        epochs=epochs,
        validation_data=val_input,
        verbose=2
    )
    plot_history(history, title_prefix="Aug(Model-internal). ", save_prefix="aug")

elif MODEL_TO_RUN == "vgg":
    model, vgg = build_vgg_model()
    adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    print("jichoi: vgg.trainable =", vgg.trainable)
    print("jichoi: trainable weights =", len(model.trainable_weights))

    history = model.fit(
        train_input,
        epochs=epochs,
        validation_data=val_input,
        verbose=2
    )
    plot_history(history, title_prefix="VGG(Model-internal aug). ", save_prefix="vgg")

else:
    raise ValueError('MODEL_TO_RUN must be one of: "baseline", "aug", "vgg"')
