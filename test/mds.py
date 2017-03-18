import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from sklearn.metrics.pairwise import cosine_similarity

class mds:

	def __init__(self, vector, ):

		pass

	def visual(vector, index, km):

		cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#0f2ce2', 6: '#f7f307', 7: '#f70707', 8: '#07f7e2'}
		
		# simi = euclidean_distances(vector)
		cluster_names = {0: 'Cluster 1', 
             1: 'Cluster 2', 
             2: 'Cluster 3', 
             3: 'Cluster 4', 
             4: 'Cluster 5',
             5: 'Cluster 6',
             6: 'Cluster 7',
             7: 'Cluster 8',
             8: 'Cluster 9'}

		mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=0,dissimilarity="precomputed", n_jobs=1)
		
		dist = 1 - cosine_similarity(vector)
		pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
		xs, ys = pos[:, 0], pos[:, 1]

		clusters = km.labels_.tolist()
		#create data frame that has the result of the MDS plus the cluster numbers and titles
		df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title = index)) 

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
		for i in range(len(df)):
			ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

		plt.show() #show the plot

		#uncomment the below to save the plot if need be
		#plt.savefig('clusters_small_noaxes.png', dpi=200)


