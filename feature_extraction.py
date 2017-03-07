import os
from sklearn.feature_extraction.text import CountVectorizer
import json

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

	corpus, index = read(dir_cur + '/doc.json')

	vectorizer = CountVectorizer(stop_words='english')
	x = vectorizer.fit_transform(corpus)

	vector = x.toarray()
	print "vector element number, should be 24:", len(vector)

	for i in xrange(len(vector)):
		print "the vector length of file", index[i], "is:", len(vector[i])

if __name__ == '__main__':
	main()