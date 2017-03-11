import numpy as np

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

	simi = euclidean_distances(vector)

	mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=0,dissimilarity="precomputed", n_jobs=1)
	pos = mds.fit(simi).embedding_

	# Rescale the data
	pos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((pos ** 2).sum())

		# Rotate the data
	clf = PCA(n_components=2)
	vector = clf.fit_transform(vector)

	pos = clf.fit_transform(pos)

if __name__ == '__main__':
	main()