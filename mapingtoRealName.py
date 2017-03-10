import os
import json

def read(fdir):

	with open(fdir) as json_data:
		json_data = json.load(json_data)
	return json_data

def fname():
	filename = {'gap_-C0BAAAAQAAJ': 'Dictionary of Greek and Roman Geography, Edited by William Smith, Vol II',
	'gap_2X5KAAAAYAAJ':'The Works of Cornelius Tacitus, by Arthur Murphy, Vol V',
	'gap_9ksIAAAAQAAJ':'The History of the Peloponnesian War, translated from the Greek of Thucydides, By William Smith, Volume the Second',
	'gap_Bdw_AAAAYAAJ' :'The History of Rome By Titus Liviusm, translated by George Baker, Vol I',
	'gap_CSEUAAAAYAAJ' :'The History of the Decline and Fall of The Roman Empire, by Edward Gibbon, Vol III',
	'gap_CnnUAAAAMAAJ':'The Whole Genuie Works of Flavius Josephus, by William Whiston, Vol II',
	'gap_DhULAAAAYAAJ':'The Description of Greece, by Pausanias, Vol III',
	'gap_DqQNAAAAYAAJ':'LIVY , History of Rome, translated by George Baker, Vol III',
	'gap_GIt0HMhqjRgC': 'Gibbon\'s History of the Decline and Fall of The Roman Empire, by Thomas Bowdler, Vol IV',
	'gap_IlUMAQAAMAAJ' : 'Gibbon\'s History of the Decline and Fall of the Roman Empire, by Thomas Bowdler, Vol II',
	'gap_MEoWAAAAYAAJ' : 'The Historical Annals of Cornelius Tacitus, by Arthur Murphy, Vol I',
	'gap_RqMNAAAAYAAJ' : 'LIVY , History of Rome, translated by George Baker, Vol V',
	'gap_TgpMAAAAYAAJ' : 'The Genuie Works of Flavius Josephus, by William Whiston, Vol I',
	'gap_VPENAAAAQAAJ' : 'The History of the Decline and Fall of The Roman Empire, by Edward Gibbon, Vol V',
	'gap_WORMAAAAYAAJ' : 'The Histories of Caius Cornelius Tacitus, by William Seymour Tyler',
	'gap_XmqHlMECi6kC' : 'The History of the Decline and Fall of The Roman Empire, by Edward Gibbon, Vol VI',
	'gap_aLcWAAAAQAAJ' : 'The History of the Decline and Fall of The Roman Empire, by Edward Gibbon, Vol I',
	'gap_dIkBAAAAQAAJ' : 'The History of Rome by Theoder Mommsen, translated by William P. Dickson, Vol III',
	'gap_fnAMAAAAYAAJ' : 'The History of the Peloponnesian War by Thucydides, By William Smith, Vol I',
	'gap_m_6B1DkImIoC' : 'Titus Livus, Roman History, by William Gordon',
	'gap_ogsNAAAAIAAJ' : 'The Works of Josephus, by William Whiston, Vol IV',
	'gap_pX5KAAAAYAAJ' : 'The Works of Cornelius Tacitus, by Arthur Murphy, Vol IV',
	'gap_udEIAAAAQAAJ' : 'The First and Thirty-Third Books of Pliny\'s Natural History, by John Bostock',
	'gap_y-AvAAAAYAAJ' : 'The Genuie Works of Flavius Josephus, by William Whiston, Vol III'}

	return filename

def main():

	dir_cur = os.getcwd() + '/gap-html'

	json_data = read(dir_cur + '/doc.json')
	
	file_name = dir_cur + '/doc_realName.json'
	dic = []
	fname = fname()

	for doc in json_data:

		# content = merge(doc['file'])
		data = {'dir': fname[doc['dir']], 'text':doc['text']}
		dic.append(data)
		print doc['dir']
	with open(file_name, 'a') as outfile:
		json.dump(dic, outfile)

if __name__ == '__main__':
	main()