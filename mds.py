import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, MeanShift
import json
from time import time
import test
import visual 

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

	k = 8
	print "Applying KMeans Clustering "	
	km = kcluster(vector, k)
	km1 = kcluster(x, k)
	order_centroids = km.cluster_centers_.argsort()[:, ::-1]

	labels = km.labels_
	
	mds = visual.mds()
	mds.visual(vector, index, km)


if __name__ == '__main__':
	main()