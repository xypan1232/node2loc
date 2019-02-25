# node2loc
To identify the functions of a protein, we first need know where this protein is located. Interacting proteins tend to locate in the same subcellular location. Thus, it is imperative to take the protein-protein interactions into account for computational identification of protein subcellular locations. <br>
we present a deep learning based method, node2loc, to predict protein subcellular location. node2loc first learns distributed representations of proteins in a protein-protein network, which acquires representations from unlabeled data for downstream tasks. Then the learned representations are further fed into a recurrent neural network (RNN) to predict subcellular locations. 

# Dependencies and development enviroment

## Package dependencies
  * <a href=https://github.com/scikit-learn/scikit-learn>sklearn 0.20.0</a> , and also its dependency numpy, pandas and scipy. <br>
  * <a href=https://github.com/scikit-learn-contrib/imbalanced-learn>imbalanced-learn</a> <br>
  * <a href=https://www.tensorflow.org/> TensorFlow 1.10+ </a> <br>
  * Python 3.6 <br>
  
## OS Requirements
This package is supported for *Linux* operating systems. The package has been tested on the following systems: <br>
Linux: Ubuntu 16.04  <br>
  
# 1. Learn node embedding from a protein-protein network using node2vec
1. Download the human protein-protein network from STRING database v9.1, and download the compressed file <a href="http://string91.embl.de/newstring_cgi/show_download_page.pl?UserId=wOOpKXCrcQGf&sessionId=fcg4u2oXFFYd">protein.links.v9.1.txt.gz</a> <br>
2. Download the node2vec software from the wbesite <a href="https://snap.stanford.edu/node2vec/">node2vec</a>. you can directly download the source code from <a href="https://github.com/aditya-grover/node2vec">node2vec github </a> in working directory. <br>
3. Run the python script to generate the node embedding: <br>
```python src/main.py --input STRING_9.1_edge.txt --output STRING_9.1_edge_500D.emd --dimensions 500```
<br>
where STRING_9.1_edge.txt is the human protein-protein network, STRING_9.1_edge_500D.emd is the learned embedding for all proteins in the network, and 500 is the specified dimension of the learned embedding. <br>
<br>
Please refer to <a href="https://github.com/aditya-grover/node2vec">node2vec github </a> for more details about how to prepare the input.<br>

### The supported input format is an edgelist: <br>
	node1_id_int node2_id_int
where node1_id_int can be the protein ID. <br>
<br>
### The output file has *n+1* lines for a graph with *n* vertices.  <br>
The first line has the following format: <br>
	num_of_nodes dim_of_representation <br>
<br>
The next *n* lines are as follows: <br>
	node_id dim1 dim2 ... dimd <br>

where dim1, ... , dimd is the *d*-dimensional representation learned by *node2vec*. <br>

# 2. Reorder the learned embedding using Minimum redundancy maximum relevance (mRMR).
1. Download the mRMR source code from the website <a href="http://home.penglab.com/proj/mRMR/index.htm"><http://home.penglab.com/proj/mRMR/index.htm </a>. <br>

# 3. Train a LSTM classifier using learned embedding, including version with Synthetic Minority Over-sampling Technique (SMOTE) and without SMOTE.

In this study, node2loc consists of the following componenets: 1) learned embedding from a protein-protein network using node2vec; 2) SMOTE for over-sampling minority classes; 3) a LSTM classifier for classifying 16 subcellular locaitons. <br>

## 3.1 Train and test LSTM classifier without SMOTE for over-sampling.
1. Train the LSTM classifier without SMOTE for over-sampling:<br>
``` python3 rnn-kfold-run.py -c 16 --datapath nitration_standard_train.csv -e 500 -u 400 -k 10``` <br>
where -c is the number of classes, --datapath is the training data with embedding as features, locaitons as the labels, -e is the dimension of embedding, -u is number of neurons in hidden layer, k is k-fold cross-validaiton. <br>
2. Train the LSTM classifier without SMOTE for over-sampling and predict subcellular locations for new proteins: <br>
```python3 rnn-pred-run.py --train nitration_standard_train.csv --test nitration_standard_test.csv``` <br>
where --train is the input training data, and --test is the input test data. <br>

## 3.2 Train and test LSTM classifier with SMOTE for over-sampling.
1. Train the LSTM classifier with SMOTE for over-sampling:<br>
``` python3 rnn-kfold-smote-run.py -c 16 --datapath nitration_standard_train.csv -e 500 -u 400 -k 10``` <br>
where -c is the number of classes, --datapath is the training data with embedding as features, locaitons as the labels, -e is the dimension of embedding, -u is number of neurons in hidden layer, k is k-fold cross-validaiton. <br>
2. Train the LSTM classifier with SMOTE for over-sampling and predict subcellular locations for new proteins: <br>
```python3 rnn-smote-pred-run.py --train nitration_standard_train.csv --test nitration_standard_test.csv``` <br>
where --train is the input training data, and --test is the input test data. <br>

