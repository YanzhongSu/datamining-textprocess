import os
import json

def read(fdir):

	with open(fdir) as json_data:
		json_data = json.load(json_data)
	return json_data

def merge(doc):

	content = ''

	for html in doc:

		content = content + ' ' + html['text']

	return content


def main():

	dir_cur = os.getcwd() + '/gap-html'

	json_data = read(dir_cur + '/text.json')
	file_name = dir_cur + '/doc.json'

	for doc in json_data:

		content = merge(doc['file'])
		data = {'dir': doc['dir'], 'text':content}

		with open(file_name, 'a') as outfile:
			json.dump(data, outfile)

if __name__ == '__main__':
	main()