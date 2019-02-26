import time
import numpy as np
from sklearn.manifold import TSNE
import pdb
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 10.0
'''
import pandas as pd

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

df = pd.DataFrame(X,columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))

X, y = None, None

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df.values)

df_tsne = df.copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        + geom_point(size=70,alpha=0.1) \
        + ggtitle("tSNE dimensions colored by digit")
chart
'''

def visualize_scatter(data_2d, label_ids, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.grid()
    
    nb_classes = len(np.unique(label_ids))
    
    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color= plt.cm.Set1(label_id / float(nb_classes)),
                    linewidth='1',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    plt.legend(loc='best')
	

#id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
labels_ss = "Biological membrane	Cell periphery	Cytoplasm	Cytoplasmic vesicle	Endoplasmic reticulum	Endosome	Extracellular space or cell surface	Flagellum or cilium	Golgi apparatus	Microtubule cytoskeleton	Mitochondrion	Nuclear periphery	Nucleolus	Nucleus	Peroxisome	Vacuole"
labels_list = labels_ss.split("\t")
id_to_label_dict = {k +1 : v for k, v in enumerate(labels_list)}
data0 = np.genfromtxt('train_dataset.csv', delimiter = ',', skip_header= 1)
label_ids =  data0[:, 0]
result = data0[:, 2:]
#pdb.set_trace()
tsne = TSNE(n_components=2, perplexity=40.0)
tsne_result = tsne.fit_transform(result)
#tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
visualize_scatter(tsne_result, label_ids)
plt.show()
