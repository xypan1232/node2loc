# -*- coding: utf-8 -*-
"""
A Recurrent Neural Network (LSTM) with SMOTE implementation based on TensorFlow and sklearn-kit library.
Author: liumin@shmtu.edu.cn
Date: 2018-10-21
Tested under: Python3.6 / TensorFlow 1.10+ / Scikit-learn 0.20.0
Derived from: Aymeric Damien
Source: https://github.com/aymericdamien/TensorFlow-Examples/
Cross-validation: k-fold using sklearn.model_selection.KFold
Source: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
Encode categorical integer features using a one-hot aka one-of-K scheme.
Source: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
imbalanced-learn
Source: https://github.com/scikit-learn-contrib/imbalanced-learn
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder  # One-hot matrix transform
from imblearn.over_sampling import SMOTE

# 避免输出TensorFlow未编译CPU指令集信息

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DISPLAY_STEP = 10


def run(inputFile, n_class, h_units, fragment, epochs, folds, l_rate, random_s=None):
    """
    RNN主程序
    参数
    ----
    inputFile: 训练集文件路径
    n_class: 分类数，即输出层单元数，默认2分类问题
    h_units: 隐藏层单元数
    fragment: 序列片段长度
    epochs: 每个fold的训练次数
    folds: k-fold折数
    l_rate: Learning rate
    random_s: 随机种子
    """
    try:
        # 导入CSV数据
        df = pd.read_csv(inputFile, encoding='utf8')
    except (OSError) as e:
        print("\n\t", e)
        print("\nPlease make sure you input correct filename of training dataset!")
        sys.exit(1)
    # 设置图级别的seed
    tf.set_random_seed(random_s)
    # 分类矩阵为第一列数据
    n_target = df.iloc[:, 0].values
    # 特征矩阵为去第一列之后数据
    n_features = df.iloc[:, 1:].values
    # 样本数
    nums_samples = n_target.size

    # 设定 K-fold 分割器
    rs = KFold(n_splits=folds, shuffle=True, random_state=random_s)

    # 整个序列长度(特征矩阵维度)
    seq_length = n_features.shape[1]

    # 如未设定隐藏层单元数则数目为整个序列长度
    if h_units == -1:
        h_units = seq_length

    # 片段分组 = 整个序列长度 / 片段长度
    group = int(seq_length / fragment)

    # 输出数据基本信息
    print("\nNumber of Samples: {0}, Length of sequence: {1}, Length of fragment: {2}, Group: {3}".format(
        nums_samples, seq_length, fragment, group))
    # 不同 Class 统计
    num_categories = np.unique(n_target).size
    sum_y = np.asarray(np.unique(n_target.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)

    # Apply SMOTE 生成 fake data
    sm = SMOTE(k_neighbors=2)
    x_resampled, y_resampled = sm.fit_sample(n_features, n_target)
    # after over sampleing 读取分类信息并返回数量
    np_resampled_y = np.asarray(np.unique(y_resampled, return_counts=True))
    df_resampled_y = pd.DataFrame(np_resampled_y.T, columns=['class', 'sum'], index=None)
    print("\nNumber of samples after over sampleing:\n{0}".format(df_resampled_y))

    # 转换原始分类矩阵为 One-hot Vector
    # reshape(-1, 1) 代表将 1行多列 转为 n行1列
    enc = OneHotEncoder(categories='auto', sparse=True, dtype=np.int)
    one_hot_mat = enc.fit(n_target.reshape(-1, 1))
    # print("\nClass Info:{0}\n".format(one_hot_mat.active_features_))
    new_target = one_hot_mat.transform(n_target.reshape(-1, 1)).toarray()  # 真实数据
    new_resampled_target = one_hot_mat.transform(
        y_resampled.reshape(-1, 1)).toarray()  # 含 fake data

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

        # 生成 k-fold 索引
        resampled_index_set = rs.split(y_resampled)
        # 初始化折数
        k_fold_step = 0
        # 暂存每次选中的测试集和对应预测结果
        test_cache = pred_cache = np.array([], dtype=np.int)
        # 暂存每次测试集特征矩阵
        x_resampled_cache = np.ones([1, seq_length])

        # 迭代训练 k-fold 交叉验证
        for train_index, test_index in resampled_index_set:
            sess.run(init)
            print("\nFold:", k_fold_step)
            # 开始每个 fold 的训练(含大量 fake data)
            for epoch in range(1, epochs + 1):
                batch_x = x_resampled[train_index]  # 特征数据用于训练
                batch_y = new_resampled_target[train_index]  # 标记结果用于验证
                batch_size = train_index.shape[0]
                # Reshape data to get N seq of N elements
                batch_x = batch_x.reshape((batch_size, fragment, group))
                _, costTrain, accTrain = sess.run(
                    [train_op, loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                # 输出训练结果
                if epoch % DISPLAY_STEP == 0 or epoch == 1:
                    print("\nTraining Epoch:", '%06d' % epoch, "Train Accuracy:", "{:.6f}".format(accTrain),
                          "Train Loss:", "{:.6f}".format(costTrain), "Train Size:", batch_size)

            # 验证测试集 (通过 index 去除 fake data)
            real_test_index = test_index[test_index < nums_samples]
            batch_test_x = x_resampled[real_test_index]
            batch_test_y = new_resampled_target[real_test_index]
            batch_test_size = len(real_test_index)
            # 暂存每次测试集特征矩阵
            x_resampled_cache = np.concatenate((x_resampled_cache, batch_test_x))
            # 测试转换
            batch_test_x = batch_test_x.reshape((batch_test_size, fragment, group))

            # 代入TensorFlow计算图验证测试集
            accTest, costTest, predVal = sess.run([accuracy, loss_op, prediction], feed_dict={
                                                  X: batch_test_x, Y: batch_test_y})

            # One-hot 矩阵转换为原始分类矩阵
            argmax_test = np.argmax(batch_test_y, axis=1)
            argmax_pred = np.argmax(predVal, axis=1)
            print("\nTest dataset Index:\n", test_index)
            # print("\nActual Values:\n", argmax_test)
            # print("\nPredicted Values:\n", argmax_pred)
            print("\nFold:", k_fold_step, "Test Accuracy:", "{:.6f}".format(
                accTest), "Test Loss:", "{:.6f}".format(costTest), "Test Size (excluded fake samples):", batch_test_size)
            # 暂存每次选中的测试集和预测结果
            test_cache = np.concatenate((test_cache, argmax_test))
            pred_cache = np.concatenate((pred_cache, argmax_pred))

            print("\n=========================================================================")
            # 每个fold训练结束后次数 +1
            k_fold_step += 1

        # 原始真实数据进行模型评估
        print("\nTest Values: '{0}-test.vals.csv'".format(inputFile))
        np.savetxt('{0}-test.vals.csv'.format(inputFile), test_cache,
                   fmt='%d', delimiter=',', header='Test Values with SMOTE')
        print("\nPredicted Values: '{0}-test.pred.csv'".format(inputFile))
        np.savetxt('{0}-test.pred.csv'.format(inputFile), pred_cache, fmt='%d',
                   delimiter=',', header='Predicted Values with SMOTE')
        from .utils import model_evaluation
        model_evaluation(n_class, test_cache, pred_cache)
