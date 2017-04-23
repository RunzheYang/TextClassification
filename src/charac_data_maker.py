# -*- coding: utf-8 -*-

import numpy as np
import csv
import json
import h5py
from lxml import etree
# from gensim.models import Word2Vec
from collections import namedtuple

TrainRecords = namedtuple('TrainRecords', ['id', 'title_content', 'label'])
TestRecords = namedtuple('TestRecords', ['id', 'title_content'])

# load word dictionary
dictionary = {}

def break2charac(sentence):
	res = []
	for i in xrange(len(sentence)):
		res.append(sentence[i])
	return res

def convert2ind(word_list):
	res = []
	for wd in word_list:
		if wd in dictionary:
			# print "--", wd
			res.append(dictionary[wd])
		else:
			# print wd
			dictionary[wd] = len(dictionary)
			res.append(dictionary[wd])
	return res, len(res)


def padding(vec_stream, length):
	blank = -1
	cur_len = len(vec_stream)
	res = vec_stream
	if cur_len == 0:
		res = [-1]
		cur_len = 1

	if cur_len < length:
		res = res + [-1]*(length - cur_len)
	else:
		res = res[0:length]
	res = np.asarray(res)
	# print res.dtype
	# print res.size
	return res
# 

def make_dataset(source, target, limits, label=None):
	
	label_dict = {}
	if label is not None:
		with open(label) as label_file:
			label_reader = csv.reader(label_file)
			for l in label_reader:
				if l[1] == 'label': continue
				label_dict[l[0]] = int(l[1])

	cnt = 0
	avg_title_len = 0
	max_title_len = 0
	avg_content_len = 0
	max_content_len = 0
	records = []

	source_file = open(source, 'r')
	for line in open(source):
		article = source_file.readline()
		article = json.loads(article)

		# get the id of the article
		art_id = article['id']

		# get the title after character segmentation
		title = article['title'].strip().lstrip().rstrip()
		title = break2charac(title)
		# print " ".join(title)

		# get the content after word segmentation
		find_text = etree.XPath("//text()")
		content = find_text(etree.HTML(article['content']))
		content = [[wd for wd in break2charac(sentence.strip().lstrip().rstrip())] for sentence in content]
		content = [wd for sec in content for wd in sec]
		# print " ".join(content)

		# convert to wordvec && record the max lenth
		# title, title_len = convert2matrix(title)
		# title, title_len = convert2stream(title)
		title, title_len = convert2ind(title)
		max_title_len = title_len if title_len > max_title_len else max_title_len
		avg_title_len = float(avg_title_len) * cnt / (cnt + 1) + float(title_len) / (cnt + 1)
		# content, content_len = convert2matrix(content)
		# content, content_len = convert2stream(content)
		content, content_len = convert2ind(content)
		max_content_len = content_len if content_len > max_content_len else max_content_len
		avg_content_len = float(avg_content_len) * cnt / (cnt + 1) + float(content_len) / (cnt + 1)

		if label is not None:
			# print label_dict[art_id]
			records.append(TrainRecords(art_id, title+content, label_dict[art_id]))
		else:
			records.append(TestRecords(art_id, title+content))

		cnt += 1
		if cnt % 1000 == 0: print cnt, "articles have been processed"
		if limits != -1 and cnt >= limits: break

	del label_dict
	
	print "max_len: ", max_title_len, max_content_len, len(dictionary)
	print "avg_len: ", avg_title_len, avg_content_len, len(dictionary)

	# del dictionary

	# pad to the same length
	for ind in xrange(len(records)):
		if ind % 1000 == 0: print ind, "matrices have been padded into the same length."
		records[ind] = records[ind]._replace(title_content=padding(records[ind].title_content, 1500))

	out_file = h5py.File(target)
	feature = np.asarray(list(map(lambda x:x.title_content, records)))
	# print feature.dtype
	
	out_file.create_dataset(
		'id', 
		data=np.asarray(list(map(lambda x:np.string_(x.id), records))))
	out_file.create_dataset(
		'feature', 
		data=feature,
		dtype=feature.dtype)
	
	if label is not None:
		out_file.create_dataset(
			'label', 
			data=np.asarray(list(map(lambda x:x.label, records))))


if __name__ == '__main__':
	make_dataset("public/test.json", "dataset/charac_test.h5", -1)
	make_dataset("public/train.json", "dataset/charac_train.h5", -1, label="public/train.csv")
	
