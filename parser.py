from bs4 import BeautifulSoup
import os
import json

def parse(dir_name):

    # dir_cur = '/Users/syz/Documents/Semester2/DataMining/individualcoursework/gap-html'
    dir_cur = os.getcwd() + '/gap-html'
    dir_cur = dir_cur +'/' + dir_name
    fnames = os.listdir(dir_cur)
    data_total = []
    for fname in fnames:
        print fname
        fdir = dir_cur + '/' + fname

        HtmlFile = open(fdir, 'r')
        source_cd = HtmlFile.read()

        soup = BeautifulSoup(source_cd, 'html.parser')
        spans = soup.find_all('span', class_ = 'ocr_cinfo')
        text = ''
        for span in spans:
            text = text + ' ' + span.get_text()
        data = {'fname': fname, 'text': text}
        data_total.append(data)

    return data_total

def main():

    # dir_cur = '/Users/syz/Documents/Semester2/DataMining/individualcoursework/gap-html'
    # dir_cur1 = '/Users/syz/Documents/Semester2/DataMining/individualcoursework'
    dir_cur = os.getcwd() + '/gap-html'
    dir_names = os.listdir(dir_cur)
    file_name = dir_cur + '/text.json'
    for dir_name in dir_names:
        print dir_name
        file_data = parse(dir_name)
        data = {'dir': dir_name, 'file':file_data}

        with open(file_name, 'a') as outfile:
          json.dump(data, outfile)

if __name__ == '__main__':
    main()