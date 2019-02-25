# -*- coding: utf-8 -*-
"""
A Recurrent Neural Network (LSTM) with SMOTE implementation based on TensorFlow and sklearn-kit library.
Author: liumin@shmtu.edu.cn
Date: 2019-01-31
Tested under: Python3.6 / TensorFlow 1.10+ / Scikit-learn 0.20.0
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder  # One-hot matrix transform
from imblearn.over_sampling import SMOTE

# 避免输出TensorFlow未编译CPU指令集信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DISPLAY_STEP = 10


def run(trainFile, testFile, h_units, fragment, epochs, l_rate, random_s):
    """
    RNN主程序
    参数
    ----
    trainFile: 训练集文件路径
    testFile: 测试集文件路径
    h_units: 隐藏层单元数
    fragment: 序列片段长度
    epochs: 每个fold的训练次数
    l_rate: Learning rate
    random_s: 随机种子
    """
    try:
        # 导入CSV数据
        df = pd.read_csv(trainFile, encoding='utf8')
        df_test = pd.read_csv(testFile, encoding='utf8')
    except (OSError) as e:
        print("\n\t", e)
        print("\nPlease make sure you input correct filename of training dataset!")
        sys.exit(1)

    # 设置图级别的seed
    tf.set_random_seed(random_s)
    # 分类矩阵为第一列数据
    y = df.iloc[:, 0].values
    # 特征矩阵为去第一列之后数据
    X = df.iloc[:, 1:].values
    # 样本数
    nums_samples = y.size
    # 读取测试集数据
    y_t = df_test.iloc[:, 0].values
    X_t = df_test.iloc[:, 1:].values
    # 测试样本数
    nums_t_samples = y_t.size

    # 整个序列长度(特征矩阵维度)
    seq_length = X.shape[1]

    # 如未设定隐藏层单元数则数目为整个序列长度
    if h_units == -1:
        h_units = seq_length

    # 片段分组 = 整个序列长度 / 片段长度
    group = int(seq_length / fragment)

    # 输出数据基本信息
    print("\nNumber of Training Samples: {0}, Length of sequence: {1}, Length of fragment: {2}, Group: {3}".format(
        nums_samples, seq_length, fragment, group))
    # 不同 Class 统计
    n_class = np.unique(y).size
    sum_y = np.asarray(np.unique(y.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)

    # Apply SMOTE 生成 fake data
    sm = SMOTE(k_neighbors=2)
    x_resampled, y_resampled = sm.fit_sample(X, y)
    # after over sampleing 读取分类信息并返回数量
    np_resampled_y = np.asarray(np.unique(y_resampled, return_counts=True))
    df_resampled_y = pd.DataFrame(np_resampled_y.T, columns=['class', 'sum'], index=None)
    print("\nNumber of training samples after over sampleing:\n{0}".format(df_resampled_y))

    # 转换原始分类矩阵为 One-hot Vector
    # reshape(-1, 1) 代表将 1行多列 转为 n行1列
    enc = OneHotEncoder(categories='auto', sparse=True, dtype=np.int)
    one_hot_mat = enc.fit(y.reshape(-1, 1))
    # print("\nClass Info:{0}\n".format(one_hot_mat.active_features_))
    new_target = one_hot_mat.transform(y.reshape(-1, 1)).toarray()  # 真实数据
    new_resampled_target = one_hot_mat.transform(
        y_resampled.reshape(-1, 1)).toarray()  # 含 fake data
    test_target = one_hot_mat.transform(y_t.reshape(-1, 1)).toarray()  # 测试集 label 转换

    # tf Graph input
    X = tf.placeholder("float", [None, fragment, group])
    Y = tf.placeholder("float", [None, n_class])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([h_units, n_class], stddev=0))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_class], stddev=0))
    }

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(X, fragment, 1)

    # Define a lstm cell with tensorflow
    # lstm_cell = rnn.BasicLSTMCell(h_units, forget_bias=1.0)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(h_units, forget_bias=1.0)

    # Get lstm cell output
    # outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # 启动会话
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(init)
        # 开始训练集训练(含大量 fake data)
        for epoch in range(1, epochs + 1):
            batch_x = x_resampled  # 特征数据用于训练
            batch_y = new_resampled_target  # 标记结果用于验证
            batch_size = batch_x.shape[0]
            # Reshape data to get N seq of N elements
            batch_x = batch_x.reshape((batch_size, fragment, group))
            _, costTrain, accTrain = sess.run(
                [train_op, loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            # 输出训练结果
            if epoch % DISPLAY_STEP == 0 or epoch == 1:
                print("\nTraining Epoch:", '%06d' % epoch, "Train Accuracy:", "{:.6f}".format(accTrain),
                      "Train Loss:", "{:.6f}".format(costTrain), "Train Size:", batch_size)
        print("\n=========================================================================")
        # 输出数据基本信息
        print("\nNumber of Test Samples: {0}, Length of sequence: {1}, Length of fragment: {2}, Group: {3}".format(
            nums_t_samples, seq_length, fragment, group))
        # 不同 Class 统计
        sum_y_t = np.asarray(np.unique(y_t.astype(int), return_counts=True))
        df_sum_y_t = pd.DataFrame(sum_y_t.T, columns=['Class', 'Sum'], index=None)
        print('\n', df_sum_y_t)
        print('\nTesting Start...')
        # 测试集数据读取开始验证
        batch_test_size = nums_t_samples
        # Reshape data to get N seq of N elements
        batch_test_x = X_t.reshape((batch_test_size, fragment, group))
        batch_test_y = test_target
        # 代入TensorFlow计算图验证测试集
        accTest, costTest, predVal = sess.run([accuracy, loss_op, prediction], feed_dict={
            X: batch_test_x, Y: batch_test_y})

        # One-hot 矩阵转换为原始分类矩阵
        argmax_test = np.argmax(batch_test_y, axis=1)
        argmax_pred = np.argmax(predVal, axis=1)
        print("\nTest Values: '{0}-test.vals.csv'".format(testFile))
        np.savetxt('{0}-test.vals.csv'.format(testFile), argmax_test,
                   fmt='%d', delimiter=',', header='Test Values with SMOTE')
        print("\nPredicted Values: '{0}-test.pred.csv'".format(testFile))
        np.savetxt('{0}-test.pred.csv'.format(testFile), argmax_pred,
                   fmt='%d', delimiter=',', header='Predicted Values with SMOTE')
        print("\nTest Accuracy:", "{:.6f}".format(accTest), "Test Loss:",
              "{:.6f}".format(costTest), "Test Size:", batch_test_size)
        print("\n=========================================================================")

        # 进行测试集模型评估
        from .utils import model_evaluation, bi_model_evaluation
        if(n_class > 2):
            model_evaluation(n_class, argmax_test, argmax_pred)
        else:
            bi_model_evaluation(argmax_test, argmax_pred)
