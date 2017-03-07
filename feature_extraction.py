import os
from sklearn.feature_extraction.text import CountVectorizer
import json
from time import time

from sklearn.cluster import KMeans

import logging

def read(fdir):
	with open(fdir) as json_data:
		json_data = json.load(json_data)

	corpus = []
	index = []
	for fitem in json_data:
		corpus.append(fitem['text'])
		index.append(fitem['dir'])

	return corpus, index

def kcluster(x):
	
	t = time()
	km = KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=1)
	
	km.fit(x)

	#print "time cost for clustering is:", time() - t
	return km

def main():

	dir_cur = os.getcwd() + '/gap-html'

	corpus, index = read(dir_cur + '/doc.json')
	t = time()
	vectorizer = CountVectorizer(stop_words='english')
	x = vectorizer.fit_transform(corpus)
	print "time cost is:", time() - t

	vector = x.toarray()
	#print "vector element number, should be 24:", len(vector)

	# for i in xrange(len(vector)):
	# 	print "the vector length of file", index[i], "is:", len(vector[i])

	# km = kcluster(x)
	

	# terms = vectorizer.get_feature_names()
	# for i in range(4):
	# 	print "Cluster :", i
	# 	for ind in order_centroids[i, :10]:
	# 		print terms[ind],
	# 	print()

if __name__ == '__main__':
	main()