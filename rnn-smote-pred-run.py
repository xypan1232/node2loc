# -*- coding: utf-8 -*-
"""
This a Basic RNN implementation with SMOTE for training and predict bio datasets.
Date: 2019-01-31
"""
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="This a Basic RNN implementation with SMOTE for training and predict bio datasets.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of training epochs.", default=20)
    parser.add_argument("-u", "--nunits", type=int,
                        help="Number of hidden layer units. (-1: use input size)", default=-1)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("-f", "--fragment", type=int,
                        help="Specifying the `length` of sequences fragment.", default=1)
    parser.add_argument("--train", type=str,
                        help="The path of training dataset.", required=True)
    parser.add_argument("--test", type=str,
                        help="The path of test dataset.", required=True)
    parser.add_argument("--learningrate", type=float,
                        help="Learning rate.", default=1e-2)
    parser.add_argument("-g", "--gpuid", type=str,
                        help='GPU to use (leave blank for CPU only)', default="")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    # logdir_base = os.getcwd()  # 获取当前目录

    # 输出RNN模型相关训练参数
    print("\nRNN HyperParameters:")
    print("\nEpochs:{0}, Learning rate:{1}, Sequences fragment length: {2}, Hidden Units:{3}".format(
        args.epochs, args.learningrate, args.fragment, args.nunits))
    print("\nRandom seed is", args.randomseed)
    # print("\nThe directory for TF logs:",
    #       os.path.join(logdir_base, args.logdir))
    print("\nGPU to use:", "No GPU support" if args.gpuid == "" else "/gpu:{0}".format(args.gpuid))
    print("\nTraining Start...")

    # 执行 RNN 训练模型并验证
    # by parsing the arguments already, we can bail out now instead of waiting
    # for TF to load, in case the arguments aren't ok
    from rnn.rnn_smote_pred import run
    run(args.train, args.test, args.nunits, args.fragment,
        args.epochs, args.learningrate, args.randomseed)
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
