import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

import os
from sklearn.feature_extraction.text import CountVectorizer
import json
from time import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MeanShift

import logging
from optparse import OptionParser
import sys

def kcluster(x, k):
	
	t = time()
	km = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state = 0)
	
	km.fit(x)

	print "time cost for clustering is:", time() - t
	return km


def read(fdir):
	with open(fdir) as json_data:
		json_data = json.load(json_data)

	corpus = []
	index = []
	for fitem in json_data:
		corpus.append(fitem['text'])
		index.append(fitem['dir'])

	return corpus, index

def main():

	dir_cur = os.getcwd() + '/gap-html'

	corpus, index = read(dir_cur + '/doc_realName.json')
	t = time()
	vectorizer = CountVectorizer(stop_words='english', min_df = 2, analyzer = 'word', token_pattern = r'\b[a-zA-Z]{4,100}\b')
	x = vectorizer.fit_transform(corpus)
	print "time cost is:", time() - t
	terms = vectorizer.get_feature_names()
	vector = x.toarray()

	# simi = euclidean_distances(vector)
	from sklearn.metrics.pairwise import cosine_similarity
	dist = 1 - cosine_similarity(vector)

	mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=0,dissimilarity="precomputed", n_jobs=1)
	
	pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

	xs, ys = pos[:, 0], pos[:, 1]
	print
	print
	k = 5
	print "Applying KMeans Clustering "	
	km = kcluster(vector, k)
	km1 = kcluster(x, k)
	order_centroids = km.cluster_centers_.argsort()[:, ::-1]

	labels = km.labels_
	clusters = km.labels_.tolist()

	#set up colors per clusters using a dict
	cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

	#set up cluster names using a dict
	cluster_names = {0: 'Family, home, war', 
                 1: 'Police, killed, murders', 
                 2: 'Father, New York, brothers', 
                 3: 'Dance, singing, love', 
                 4: 'Killed, soldiers, captain'}

	#some ipython magic to show the matplotlib plots inline
	# %matplotlib inline 


	import pandas as pd
	#create data frame that has the result of the MDS plus the cluster numbers and titles
	df = pd.DataFrame(dict(x=xs, y=ys, label=clusters)) 

	#group by cluster
	groups = df.groupby('label')

	# set up plot
	fig, ax = plt.subplots(figsize=(17, 9)) # set size
	ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

	#iterate through groups to layer the plot
	#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
	for name, group in groups:
		ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
			label=cluster_names[name], color=cluster_colors[name], 
			mec='none')
		ax.set_aspect('auto')
		ax.tick_params(\
			axis= 'x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom='off',      # ticks along the bottom edge are off
			top='off',         # ticks along the top edge are off
			labelbottom='off')
		ax.tick_params(\
			axis= 'y',         # changes apply to the y-axis
			which='both',      # both major and minor ticks are affected
			left='off',      # ticks along the bottom edge are off
			top='off',         # ticks along the top edge are off
			labelleft='off')

	ax.legend(numpoints=1)  #show legend with only 1 point

	#add label in x,y position with the label as the film title
	# for i in range(len(df)):
	# 	ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

	plt.show() #show the plot

	#uncomment the below to save the plot if need be
	#plt.savefig('clusters_small_noaxes.png', dpi=200)



	# pos = mds.fit(simi).embedding_

	# # Rescale the data
	# pos *= np.sqrt((vector ** 2).sum()) / np.sqrt((pos ** 2).sum())

	# 	# Rotate the data
	# clf = PCA(n_components=2)
	# vector = clf.fit_transform(vector)

	# pos = clf.fit_transform(pos)

	# fig = plt.figure(1)
	# ax = plt.axes([0., 0., 1., 1.])
	# plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', s= 100, lw=0, label='MDS')

	# # Plot the edges
	# start_idx, end_idx = np.where(pos)

	# segments = [[vector[i, :], vector[j, :]]
	# 	for i in range(len(pos)) for j in range(len(pos))]

	# values = np.abs(simi)
	# lc = LineCollection(segments,zorder=0, cmap=plt.cm.Blues,norm=plt.Normalize(0, values.max()))
	# lc.set_array(simi.flatten())
	# lc.set_linewidths(0.5 * np.ones(len(segments)))
	# ax.add_collection(lc)

	# plt.show()

if __name__ == '__main__':
	main()