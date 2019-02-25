# -*- coding: utf-8 -*-
"""
This a Basic RNN implementation with SMOTE for training bio datasets.
Date: 2018-10-21
"""
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="This a Basic RNN implementation with SMOTE for training bio datasets.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--nclass", type=int,
                        help="Number of class. Must be at least 2 aka two-classification.", default=2)
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of training epochs.", default=20)
    parser.add_argument("-u", "--nunits", type=int,
                        help="Number of hidden layer units. (-1: use input size)", default=-1)
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("-f", "--fragment", type=int,
                        help="Specifying the `length` of sequences fragment.", default=1)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)
    parser.add_argument("--learningrate", type=float,
                        help="Learning rate.", default=1e-2)
    parser.add_argument("-g", "--gpuid", type=str,
                        help='GPU to use (leave blank for CPU only)', default="")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    # logdir_base = os.getcwd()  # 获取当前目录

    # 输出RNN模型相关训练参数
    print("\nRNN HyperParameters:")
    print("\nN-classes:{0}, Training epochs:{1}, Learning rate:{2}, Sequences fragment length: {3}, Hidden Units:{4}".format(
        args.nclass, args.epochs, args.learningrate, args.fragment, args.nunits))
    print("\nCross-validation info:")
    print("\nK-fold:", args.kfolds, ", Random seed is", args.randomseed)
    # print("\nThe directory for TF logs:",
    #       os.path.join(logdir_base, args.logdir))
    print("\nGPU to use:", "No GPU support" if args.gpuid == "" else "/gpu:{0}".format(args.gpuid))
    print("\nTraining Start...")

    # 执行 RNN 训练模型并验证
    # by parsing the arguments already, we can bail out now instead of waiting
    # for TF to load, in case the arguments aren't ok
    from rnn.rnn_smote import run
    # from rnn.rnn_bio import run
    run(args.datapath, args.nclass, args.nunits, args.fragment,
        args.epochs, args.kfolds, args.learningrate, args.randomseed)
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
