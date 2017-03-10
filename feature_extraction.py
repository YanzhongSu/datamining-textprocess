#from __future__ import print_function
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

import numpy as np

def read(fdir):
	with open(fdir) as json_data:
		json_data = json.load(json_data)

	corpus = []
	index = []
	for fitem in json_data:
		corpus.append(fitem['text'])
		index.append(fitem['dir'])

	return corpus, index

def kcluster(x, k):
	
	t = time()
	km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
	
	km.fit(x)

	print "time cost for clustering is:", time() - t
	return km

def mCluster(x):
	t = time()
	ms = MeanShift()
	ms.fit(x)
	print "time cost for clustering is:", time() - t
	return ms

def main():

	dir_cur = os.getcwd() + '/gap-html'

	corpus, index = read(dir_cur + '/doc_realName.json')
	t = time()
	vectorizer = CountVectorizer(stop_words='english', min_df = 2, analyzer = 'word', max_df = 5)
	x = vectorizer.fit_transform(corpus)
	print "time cost is:", time() - t
	terms = vectorizer.get_feature_names()
	vector = x.toarray()
	print "vector element number, should be 24:", len(vector)

	for i in xrange(len(vector)):
		print "the vector length of file", index[i], "is:", len(vector[i])
	
	print "Applying Hierarchical Clustering "
	
	ms = mCluster(vector)
	labels = ms.labels_
	k = len(np.unique(labels))
	print "There are in total", k, "Clustering"

	cat = []
	for i in range(k):
		cat.append([])

	# mapping correspoding doc to corresponding clusters
	for i in range(len(index)):
		cat[labels[i]].append([index[i], labels[i]])
	
	order_centroids = ms.cluster_centers_.argsort()[:, ::-1]
	print "Top terms per cluster:"
	for i in range(k):
		print "Cluster :", i, "has", len(cat[i]), "documents"
		for ind in order_centroids[i, :10]:
			print terms[ind],
		print()

		for item in cat[i]:
			print item[0]
			print ()

	print ()
	print "Applying KMeans Clustering "

	for ki in range(2, 3):
		km = kcluster(vector, k)
		km1 = kcluster(x, k)
		order_centroids = km.cluster_centers_.argsort()[:, ::-1]

		labels = km.labels_

		cat = []
		for i in range(k):
			cat.append([])

		for i in range(len(index)):
			cat[labels[i]].append([index[i], labels[i]])

		print "Top terms per cluster:"
		for i in range(k):
			print "Cluster :", i, "has", len(cat[i]), "documents"
			for ind in order_centroids[i, :10]:
				print terms[ind],
			print()

			for item in cat[i]:
				print item[0]

		# print "labels"


		# 	#print index[i], ":", labels[i]

		# for item in cat:

		# 	if len(item) > 0:
		# 		print "Labels:", item[0][1], "has", len(item), "documents"

		# 	for i in item:
		# 		print i[0], ":", i[1]

if __name__ == '__main__':
	main()