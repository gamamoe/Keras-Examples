################################################################################

# MNIST 데이터 관련 import
from keras.datasets import mnist                  # MNIST 데이터 Loader
from keras.utils.np_utils import to_categorical   # One-hot 포맷 변환
import numpy as np                                # float type casting

# Feature scaling 관련 import
from sklearn.preprocessing import minmax_scale    # [0-1] Scaling

# Model 구축 관련 import
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import adam

################################################################################

# 데이터 Load 및 전처리 과정

# Train, Test 데이터 Load
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Train 데이터 포맷 변환
# 60000(Train Sample 수) * 28(가로) * 28(세로) 포맷을
# 60000(Train Sample 수) * 784(= 28 * 28) 포맷으로 수정
num_of_train_samples = X_train.shape[0]                     # Train Sample 수
width = X_train.shape[1]                                    # 가로 길이
height = X_train.shape[2]                                   # 세로 길이
X_train = X_train.reshape(num_of_train_samples, width * height)

# Test 데이터 포맷 변환
# width, height는 Train 데이터와 같으므로 재사용
# 10000(Test Sample 수) * 28(가로) * 28(세로) 포맷을
# 10000(Test Sample 수) * 784(= 28 * 28) 포맷으로 수정
num_of_test_samples = X_test.shape[0]  # Sample 수
X_test = X_test.reshape(num_of_test_samples, width * height)

# Feature Scaling
# X_train의 각 원소는 0-255 사이의 값을 가지고 있다
# Overfitting 방지 및 Cost 함수의 빠른 수렴을 위해서 
# Feature Scaling 작업을 한다.
# 예제에서는 0-255 범위를 0-1 범위로 Scaling
# 참고: https://en.wikipedia.org/wiki/Feature_scaling

# 나누기 연산이 들어가므로 uint8을 float64로 변환한다
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

# 간단한 방법은 MNIST가 0-255 사이 값만을 가진다는 것을 알기 때문에
# 단순히 255를 나눠도 Feature Scaling이 가능하다.
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# 아래 방법은 다소 복잡하지만 다른 데이터에서도 동일하게 적용할 수 있음
# Sample by featre matrix 형태이므로 axis=0로 설정
# axis=1은 축을 바꿔서 scaling, 자세한 내용은 scikit 문서 참조
X_train = minmax_scale(X_train, feature_range=(0, 1), axis=0)
X_test = minmax_scale(X_test, feature_range=(0, 1), axis=0)

# Lable의 categorical 값을 One-hot 형태로 변환
# 예를 들어 [1, 3, 2, 0] 를
# [[ 0.,  1.,  0.,  0.],
#  [ 0.,  0.,  0.,  1.],
#  [ 0.,  0.,  1.,  0.],
#  [ 1.,  0.,  0.,  0.]]
# 로 변환하는 것을 One-hot 형태라고 함
# MNIST Label인 0 ~ 9사이의 10가지 값을 변환한다.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

################################################################################
# Multilayer Perceptron (MLP) 생성

model = Sequential()

# Dense(256)의 의미는 256개의 hidden unit을 가지는 fully connected layer
# keras에서는 첫 번째 Layer, 즉 input layer의 input dimension만 지정하면
# 뒤의 연결되는 Layer의 dimension은 간단하게 작성 가능하다.

# width * height = 784인 dimension
# glorot_uniform == Xavier Initialization, keras에서는 내부적으로 이미 제공
# 그 외 he_uniform 등도 이미 구현되어있다.

# 첫 번째 Layer (Input layer)
model.add(Dense(256, input_dim=width * height, init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))        # 30% 정도를 Drop

# 두 번째 Layer (Hidden layer 1)
# 두 번째 Layer부터는 output_dim만 설정하면 된다
# input_dim은 이전 레이어의 output_dim과 같다고 가정함
model.add(Dense(256, init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

# 세 번째 Layer (Hidden layer 2)
model.add(Dense(256, init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

# 네 번째 Layer (Hidden layer 3)
model.add(Dense(256, init='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))

# 다섯 번째 Layer (Output layer)
# Output layer는 softmax activation function
number_of_class = 10  # MNIST 예제는 10가지의 Category를 가지고 있다.
model.add(Dense(number_of_class, activation='softmax'))  

# Cost function 및 Optimizer 설정
# Multiclass 분류이므로 Cross-entropy 사용
# Adam optimizer 사용
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model training
training_epochs = 15
batch_size = 100
model.fit(X_train, y_train, nb_epoch=training_epochs, batch_size=batch_size)

# Model evaluation using test set
print('모델 평가')
evaluation = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Accuracy: ' + str(evaluation[1]))
