# -*- coding: utf-8 -*-
"""
A Recurrent Neural Network (LSTM) implementation example using TensorFlow and sklearn-kit library.
Author: liumin@shmtu.edu.cn
Date: 2017-12-7
Tested under: Python3.5 / Python3.6 and TensorFlow 1.1 / 1.2 / 1.3
Derived from: Aymeric Damien
Source: https://github.com/aymericdamien/TensorFlow-Examples/
Cross-validation: k-fold using sklearn.model_selection.KFold
Source: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
Encode categorical integer features using a one-hot aka one-of-K scheme.
Source: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder  # One-hot matrix transform

# 避免输出TensorFlow未编译CPU指令集信息

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
        # TRAIN_CSV = os.path.join(os.path.dirname(__file__), inputFile)
        TRAIN_CSV = inputFile
        # 去掉CSV文件标题行
        train_set = np.genfromtxt(TRAIN_CSV, delimiter=',', skip_header=1)
    except (OSError) as e:
        print("\n\t", e)
        print("\nPlease make sure you input correct filename of training dataset!")
        sys.exit(1)

    # 设定 K-fold 分割器
    rs = KFold(n_splits=folds, shuffle=True, random_state=random_s)

    # 整个序列长度
    seq_length = train_set.shape[1] - 1

    # 片段分组 = 整个序列长度 / 片段长度
    group = int(seq_length / fragment)

    # 如未设定隐藏层单元数则数目为整个序列长度
    if h_units == -1:
        h_units = seq_length

    # 分类矩阵为第一列数据
    n_target = train_set[:, 0]

    # 特征矩阵为去第一列之后数据
    n_features = train_set[:, 1:]
    # 样本数
    nums_samples = n_features.shape[0]

    # 转换原始分类矩阵为 One-hot Vector
    # reshape(-1, 1) 代表将 1行多列 转为 n行1列
    enc = OneHotEncoder(categories='auto', sparse=True, dtype=np.int)
    one_hot_mat = enc.fit(n_target.reshape(-1, 1))
    new_target = one_hot_mat.transform(n_target.reshape(-1, 1)).toarray()

    # 输出样本数
    print("Number of Samples: {0}, Length of sequence: {1}, Length of fragment: {2}, Group: {3}".format(
        nums_samples, seq_length, fragment, group))
    # 不同 Class 统计
    num_categories = np.unique(n_target).size
    sum_y = np.asarray(np.unique(n_target.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)

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
    lstm_cell = tf.nn.rnn_cell.LSTMCell(h_units, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

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
    with tf.Session() as sess:

        # 生成 k-fold 训练集、测试集索引
        cv_index_set = rs.split(new_target)
        # training_step = 1  # 初始化训练次数
        k_fold_step = 1  # 初始化折数

        # 暂存每次选中的测试集和对应预测结果
        test_cache = pred_cache = np.array([], dtype=np.int)

        # 迭代训练 k-fold 交叉验证
        for train_index, test_index in cv_index_set:
            sess.run(init)
            print("\nFold:", k_fold_step)
            # print("\nTrain-index:\n", train_index, "\nTest-index:\n", test_index)
            # 开始每个 fold 的训练
            for epoch in range(1, epochs + 1):
                batch_x = n_features[train_index]  # 特征数据用于训练
                batch_y = new_target[train_index]  # 标记结果用于验证
                batch_size = train_index.shape[0]
                # Reshape data to get N seq of N elements
                batch_x = batch_x.reshape((batch_size, fragment, group))
                _, costTrain, accTrain = sess.run(
                    [train_op, loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                # 输出训练结果
                if epoch % DISPLAY_STEP == 0 or epoch == 1:
                    print("\nTraining Epoch:", '%06d' % epoch, "Train Accuracy:", "{:.6f}".format(accTrain),
                          "Train Loss:", "{:.6f}".format(costTrain), "Train Size:", batch_size)

            # 输入测试数据
            batch_test_x = n_features[test_index]
            batch_test_y = new_target[test_index]
            batch_test_size = test_index.shape[0]
            batch_test_x = batch_test_x.reshape(
                (batch_test_size, fragment, group))

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
                accTest), "Test Loss:", "{:.6f}".format(costTest), "Test Size:", batch_test_size)
            # 暂存每次选中的测试集和预测结果
            test_cache = np.concatenate((test_cache, argmax_test))
            pred_cache = np.concatenate((pred_cache, argmax_pred))

            print(
                "\n=========================================================================")
            # 每个fold训练结束后次数 +1
            k_fold_step += 1

        # 模型评估结果输出
        print("\nTest Values: '{0}-test.vals.csv'".format(inputFile))
        np.savetxt('{0}-test.vals.csv'.format(inputFile), test_cache,
                   fmt='%d', delimiter=',', header='Test Values without SMOTE')
        print("\nPredicted Values: '{0}-test.pred.csv'".format(inputFile))
        np.savetxt('{0}-test.pred.csv'.format(inputFile), pred_cache, fmt='%d',
                   delimiter=',', header='Predicted Values without SMOTE')
        from .utils import model_evaluation
        model_evaluation(n_class, test_cache, pred_cache)
