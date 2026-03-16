import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 지저분한 C++ 경고 끄기
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # WSL2 CUDNN 에러를 막는 핵심: GPU 메모리를 필요한 만큼만 천천히 할당하도록 설정
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print('\n🎉 대성공! GPU가 완벽하게 인식되고 메모리 세팅도 끝났습니다.')
        print('인식된 장치:', gpus[0])
    except RuntimeError as e:
        print('\n❌ 장치는 찾았지만 세팅 중 에러 발생:', e)
else:
    print('\n❌ 실패... GPU를 찾지 못했습니다.')
    