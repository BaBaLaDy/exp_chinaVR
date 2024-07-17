import sys
import warnings
import os
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from comm import normalize_fun, csv_fun
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import preprocessing

# data_floder = 'E:/unity_pro/My_workV0.3/Assets/StreamingAssets/SourcesFolder/Data_Process/'
# model_floder = './model/model_output/'
data_floder = os.getcwd()
data_floder = os.path.abspath(os.path.join(data_floder, 'Assets', "StreamingAssets\SourcesFolder\Data_Process"))
data_floder += '/'
model_floder = os.getcwd()
model_floder = os.path.abspath(os.path.join(model_floder, 'Assets', "Scrips\Work\py\model\model_output"))
model_floder += '/'

validation_split = 0.20
verbose = 0  # 不输出epoch进度条

# _NN_model.h5
# _CNN_model.h5

use_2d = 1


def model_finetune(object_model, data_source, loss, epoch_num):
    (X_train, X_test, Y_train, Y_test, labels) = get_data(data_source=data_source, object_model=object_model)
    model_path = model_floder + "_" + object_model + "_model.h5"
    if use_2d and object_model == 'CNN':
        model_path = model_floder + "_" + object_model + "2D" + "_model.h5"
    model = tf.keras.models.load_model(model_path)

    for i in range(len(model.layers) - 1):
        model.layers[i].trainable = False
    # compile
    model.compile(optimizer='adam', loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    # fit
    model.fit(X_train, Y_train, epochs=epoch_num, validation_split=validation_split, verbose=verbose)

    # model save
    model.save(model_path)

    # 转换模型
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # open("CNN_Model.tflite", "wb").write(tflite_model)
    # # Load TFLite model and allocate tensors.
    # interpreter = tf.lite.Interpreter(model_path="CNN_Model.tflite")
    # interpreter.allocate_tensors()
    #
    # # Get input and output tensors.
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    #
    # print(input_details)
    # print(output_details)

    # print message
    if verbose:
        print(len(X_train), len(X_test), len(Y_train), len(Y_test))
        print(len(model.layers))
        model.summary()

    # model evaluate
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=verbose)
    print('\nTest accuracy:', test_acc)

    # predit
    y_predit_label = []
    y_predit_val = model.predict(X_test, verbose=verbose)
    y_predit_index = np.argmax(y_predit_val, axis=1)
    for i in range(len(y_predit_index)):
        label = labels[y_predit_index[i]]
        y_predit_label.append(label)

    return y_predit_label


def get_data(data_source, object_model):
    '''
    函数功能：获取训练集与测试集
    :return: 无
    '''
    # 读取数据
    if (data_source == 'r'):
        train_file_path = data_floder + 'imureal_all_train_data.csv'
        test_file_path = data_floder + 'imureal_all_test_data.csv'
    elif (data_source == 'e'):
        train_file_path = data_floder + 'imureal_all_train_data_e.csv'
        test_file_path = data_floder + 'imureal_all_test_data_e.csv'
    file_header = csv_fun.get_csv_header(train_file_path)
    feature_names = file_header[1:len(file_header)]
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    # get labels names
    labels = np.unique(test_data['label'].values)
    labels = sorted(labels, key=str.lower)

    # 文本标签转化为数字标签
    le = preprocessing.LabelEncoder()  # 获取一个LabelEncoder
    train_data['label'] = le.fit_transform(train_data['label'])  # 使用训练好的LabelEncoder对原数据进行编码
    test_data['label'] = le.fit_transform(test_data['label'])  # 使用训练好的LabelEncoder对原数据进行编码
    train_data_len = train_data.shape[0]
    test_data_len = test_data.shape[0]
    column_num = train_data.shape[1]
    label_num = 1
    features_num = column_num - label_num

    # 得到训练集与测试集
    X_train = train_data.iloc[:, 1:column_num].values.reshape(
        (train_data_len, features_num))
    Y_train = train_data.iloc[:, 0:label_num].values.reshape((train_data_len, label_num))
    X_test = test_data.iloc[:, 1:column_num].values.reshape((test_data_len, features_num))
    Y_test = test_data.iloc[:, 0:label_num].values.reshape((test_data_len, label_num))

    if use_2d and object_model == 'CNN':  # 使用conv2d
        # 将数据1X120转成3X40
        X_train = X_train.reshape((train_data_len, int(features_num / 3), 3))
        X_test = X_test.reshape((test_data_len, int(features_num / 3), 3))

        # 对数据进行归一化处理
        X_train = normalize_fun.normalize_3dimensions(X_train)
        X_test = normalize_fun.normalize_3dimensions(X_test)

        # 增加灰度维度防止conv2d报错
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
    else:
        # 对数据进行归一化处理
        X_train, features_mean, features_deviation = normalize_fun.normalize(X_train)
        X_test = normalize_fun.normalize_designate_paras(features=X_test,
                                                         features_mean=features_mean,
                                                         features_deviation=features_deviation)

        # 保存归一化参数
        csv_header = []
        for i in range(features_num):
            csv_header.append(i)
        csv_fun.csv_init(file_path=data_floder + 'features_mean.csv', csv_head=csv_header)
        csv_fun.csv_append_data(file_path=data_floder + 'features_mean.csv', data_row=features_mean)
        csv_fun.csv_init(file_path=data_floder + 'features_deviation.csv', csv_head=csv_header)
        csv_fun.csv_append_data(file_path=data_floder + 'features_deviation.csv', data_row=features_deviation)

    return X_train, X_test, Y_train, Y_test, labels


def model_fine_tune():
    object_model = sys.argv[1]
    data_source = sys.argv[2]
    loss = sys.argv[3]
    epoch_num = 50 if sys.argv[4] == '0' else int(sys.argv[4])
    print('object_model=', object_model,
          'data_source=', data_source,
          'loss=', loss,
          'epoch_num=', epoch_num)
    model_finetune(object_model=object_model,
                   data_source=data_source,
                   loss=loss,
                   epoch_num=epoch_num)


# 单独运行Python时用
# predit = model_finetune('CNN', "r", "SparseCategoricalCrossentropy", 50)
# print(predit)

# Unity与Python联合运行时用
model_fine_tune()

