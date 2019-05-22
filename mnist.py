# coding=utf-8
# 导入相应的库
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils


mnist_train = pd.read_csv('../mnist/train.csv')  # 训练集
mnist_test = pd.read_csv('../mnist/test.csv')  # 测试集

y_train = mnist_train["label"]  # 训练集标签
y_train = np_utils.to_categorical(y_train, num_classes=10)  # 训练集标签转化
x_train = mnist_train.drop(labels=["label"], axis=1)  # DataFrame删除"label"列

x_train = x_train / 255.0  # 归一化
x_test = mnist_test / 255.0

x_train = x_train.values.reshape(-1, 28, 28, 1)
x_test = x_test.values.reshape(-1, 28, 28, 1)


# cnn模型
def cnn_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same', input_shape=(28, 28, 1)))  # 卷积层
    model.add(BatchNormalization())  # 批量归一化层
    model.add(Activation(activation='relu'))  # 激活层
    model.add(Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))  # 最大池化层

    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())  # 平铺
    model.add(Dense(128))  # 全连接层
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation(activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])

    return model


# 训练模型
def model_fit(model, x, y):
    return model.fit(x, y, batch_size=100, epochs=10, verbose=1, validation_split=0.2)


# 预测
def model_predict(model, x):
    return model.predict_classes(x)


model = cnn_model()
train_model = model_fit(model, x_train, y_train)

# 预测
y_pred = model_predict(model, x_test)

sub = pd.read_csv('../mnist/sample_submission.csv')
sub['Label'] = y_pred

# 保存结果
sub.to_csv('../mnist/my_submission.csv', index=False)
