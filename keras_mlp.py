from keras.datasets import mnist                  # MNIST 데이터 Loader
from keras.utils.np_utils import to_categorical   # One-hot 포맷 변환
import numpy as np                                # float type casting

from sklearn.preprocessing import minmax_scale    # [0-1] Scaling

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import adam

# 데이터 Load 및 전처리 과정

# Train, Test 데이터 Load
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Train 데이터 포맷 변환
num_of_train_samples = X_train.shape[0]                     # Train Sample 수
width = X_train.shape[1]                                    # 가로 길이
height = X_train.shape[2]                                   # 세로 길이
X_train = X_train.reshape(num_of_train_samples, width * height)

# Test 데이터 포맷 변환
num_of_test_samples = X_test.shape[0]  # Sample 수
X_test = X_test.reshape(num_of_test_samples, width * height)

# Feature Scaling
# 나누기 연산이 들어가므로 uint8을 float64로 변환한다
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

X_train = minmax_scale(X_train, feature_range=(0, 1), axis=0)
X_test = minmax_scale(X_test, feature_range=(0, 1), axis=0)

# MNIST Label인 0 ~ 9사이의 10가지 값을 변환한다.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Multilayer Perceptron (MLP) 생성
model = Sequential()

# width * height = 784인 dimension
# glorot_uniform == Xavier Initialization, keras에서는 내부적으로 이미 제공

model.add(Dense(256, input_dim=width * height, init='glorot_uniform'))
model.add(Activation('relu'))  # Activation 함수로 relu 사용
model.add(Dropout(0.3))        # 30% 정도를 Drop

# 두 번째 Layer부터는 output_dim만 설정하면 된다
# input_dim은 이전 레이어의 output_dim과 같다고 가정함
model.add(Dense(256, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.3))

# 세 번째 Layer (Hidden layer 2)
model.add(Dense(256, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.3))

# 네 번째 Layer (Hidden layer 3)
model.add(Dense(256, init='glorot_uniform'))
model.add(Activation('relu'))
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
