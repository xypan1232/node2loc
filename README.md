# node2loc
it first uses node2vec to learn node embedding of proteins from a interaction network, then the learned embedding is fed into a recurrenct neural network for classifying protein subcellular locations.

## Package dependencies
  * <a href=https://github.com/scikit-learn/scikit-learn>sklearn</a> , and also its dependency numpy, pandas and scipy. <br>
  * <a href=https://github.com/scikit-learn-contrib/imbalanced-learn>imbalanced-learn</a> <br>
  * <a href=https://www.tensorflow.org/> Tensorflow </a> <br>
  * Python 3.6 <br>

  
## Learn node embedding from a protein-protein network using node2vec
1. Download the human protein-protein network from STRING database v9.1, and download the compressed file <a href="http://string91.embl.de/newstring_cgi/show_download_page.pl?UserId=wOOpKXCrcQGf&sessionId=fcg4u2oXFFYd">protein.links.v9.1.txt.gz</a> <br>
2. Download the node2vec software from the wbesite <a href="https://snap.stanford.edu/node2vec/">node2vec</a>. <br>
3. Run the python script to generate the node embedding: <br>
```python src/main.py --input STRING_9.1_edge.txt --output STRING_9.1_edge_500D.emd --dimensions 500```
<br>
where STRING_9.1_edge.txt is the human protein-protein network, STRING_9.1_edge_500D.emd is the learned embedding for all proteins in the network, and 500 is the specified dimension of the learned embedding. <br>

## Reorder the learned embedding using Minimum redundancy maximum relevance (mRMR).
1. Download the mRMR source code from the website <a href="http://home.penglab.com/proj/mRMR/index.htm"><http://home.penglab.com/proj/mRMR/index.htm </a>. <br>

## train a LSTM classifier using learned embedding
