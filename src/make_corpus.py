import json
import jieba
from lxml import etree

# load stopping words
stp_wd = []
with open("support_data/stopwd.txt", 'r') as stp_file:
	stp_wd = stp_file.readlines()
	# convert to utf-8 and delete '\n' at the end
	stp_wd = [wd.decode('utf-8')[:-1] for wd in stp_wd]

def censor(text):
	text_censored = []
	for wd in text:
		if wd not in stp_wd:
			text_censored.append(wd)
	return text_censored

def make_corpus(source, target, limits):
	cnt = 0
	file = open(source, 'r')
	for line in open(source):
		article = file.readline()
		article = json.loads(article)

		# get the title after word segmentation
		title = jieba.cut(article['title'])
		title = [wd for wd in title]

		# get the content after word segmentation
		find_text = etree.XPath("//text()")
		content = find_text(etree.HTML(article['content']))
		content = [[wd for wd in jieba.cut(sentence.strip().lstrip().rstrip())] for sentence in content]
		content = [wd for sec in content for wd in sec]

		# print "".join(title)
		# print "".join(content)

		# Question: should we delete stopping words?
		# if yes:
		title = censor(title)
		content = censor(content)

		# dumps to target file
		out_file = open(target, 'a')
		print >> out_file, u" ".join(title).encode('utf-8'), u" ".join(content).encode('utf-8')

		cnt += 1
		if cnt % 1000 == 0: print cnt, "articles have been processed"
		if limits != -1 and cnt >= limits: break

make_corpus("public/train.json", "support_data/corpus.txt", -1)
make_corpus("public/test.json", "support_data/corpus.txt", -1)
