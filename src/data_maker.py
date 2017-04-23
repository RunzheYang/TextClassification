# -*- coding: utf-8 -*-

import numpy as np
import csv
import json
import h5py
import jieba
from lxml import etree
# from gensim.models import Word2Vec
from collections import namedtuple

TrainRecords = namedtuple('TrainRecords', ['id', 'title', 'content', 'label'])
TestRecords = namedtuple('TestRecords', ['id', 'title', 'content'])

# load stopping words
stp_wd = []
with open("support_data/stopwd.txt", 'r') as stp_file:
	stp_wd = stp_file.readlines()
	# convert to utf-8 and delete '\n' at the end
	stp_wd = [wd.decode('utf-8')[:-1] for wd in stp_wd]

# wv_model = Word2Vec.load('support_data/corpus2vec.model')

# load word dictionary
dictionary = {}
with open("support_data/word_index.txt", 'r') as dic_file:
	for line in open("support_data/word_index.txt"):
		rec = dic_file.readline()
		rec = rec[:-1]
		rec = rec.split(' ')
		dictionary[rec[1].decode('utf-8')] = int(rec[0])


def censor(text):
	text_censored = []
	for wd in text:
		if wd not in stp_wd:
			text_censored.append(wd)
	return text_censored


# def convert2matrix(text):
# 	res = []
# 	for wd in text:
# 		# if wd in wv_model:
# 		# 	res.append(wv_model[wd])
# 		if wd in 
# 	return res, len(res)


# def convert2stream(word_list):
	# res = []
	# for wd in word_list:
	# 	if wd in wv_model:
	# 		res.append(np.string_(wd.encode('utf-8')))
	# return res, len(res)

def convert2ind(word_list):
	res = []
	for wd in word_list:
		if wd in dictionary:
			# print wd
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
	max_title_len = 0
	max_content_len = 0
	records = []

	source_file = open(source, 'r')
	for line in open(source):
		article = source_file.readline()
		article = json.loads(article)

		# get the id of the article
		art_id = article['id']

		# get the title after word segmentation
		title = jieba.cut(article['title'])
		title = [wd for wd in title]

		# get the content after word segmentation
		find_text = etree.XPath("//text()")
		content = find_text(etree.HTML(article['content']))
		content = [[wd for wd in jieba.cut(sentence.strip().lstrip().rstrip())] for sentence in content]
		content = [wd for sec in content for wd in sec]

		# Question: should we delete stopping words?
		# if yes:
		title = censor(title)
		content = censor(content)

		# convert to wordvec && record the max lenth
		# title, title_len = convert2matrix(title)
		# title, title_len = convert2stream(title)
		title, title_len = convert2ind(title)
		max_title_len = title_len if title_len > max_title_len else max_title_len
		# content, content_len = convert2matrix(content)
		# content, content_len = convert2stream(content)
		content, content_len = convert2ind(content)
		max_content_len = content_len if content_len > max_content_len else max_content_len

		if label is not None:
			records.append(TrainRecords(art_id, title, content, label_dict[art_id]))
		else:
			records.append(TestRecords(art_id, title, content))

		cnt += 1
		if cnt % 1000 == 0: print cnt, "articles have been processed"
		if limits != -1 and cnt >= limits: break

	del label_dict
	
	print "max_len: ", max_title_len, max_content_len

	# pad to the same length
	for ind in xrange(len(records)):
		if ind % 1000 == 0: print ind, "matrices have been padded into the same length."
		records[ind] = records[ind]._replace(title=padding(records[ind].title, 20))
		records[ind] = records[ind]._replace(content=padding(records[ind].content, 500))

	out_file = h5py.File(target)
	title_feature = np.asarray(list(map(lambda x:x.title, records)))
	# print title_feature.dtype
	content_feature = np.asarray(list(map(lambda x:x.content, records)))
	# print content_feature.dtype
	
	out_file.create_dataset(
		'id', 
		data=np.asarray(list(map(lambda x:np.string_(x.id), records))))
	out_file.create_dataset(
		'title_feature', 
		data=title_feature,
		dtype=title_feature.dtype)
	out_file.create_dataset(
		'content_feature',
		data=content_feature,
		dtype=content_feature.dtype)
	
	if label is not None:
		out_file.create_dataset(
			'label', 
			data=np.asarray(list(map(lambda x:x.label, records))))


if __name__ == '__main__':
	# make_dataset("public/test.json", "dataset/test.h5", -1)
	make_dataset("public/train.json", "dataset/train.h5", -1, label="public/train.csv")
	
