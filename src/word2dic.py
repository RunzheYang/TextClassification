# word2dic for feasible memory solution

from gensim.models import Word2Vec
wv_model = Word2Vec.load("support_data/corpus2vec.model")

corpus = "support_data/corpus.txt"
file = open(corpus, 'r')

word_dict = []
word_dict_check = {}
wd_count = 0

proc_cnt = 0

for line in open(corpus):
	proc_cnt += 1
	if proc_cnt % 1000 == 0: print proc_cnt, "articles have been processed."
	article = file.readline()
	article = article[:-1]
	wd_stream = article.split(' ')
	for wd in wd_stream:
		# print wd
		wd = wd.decode('utf-8')
		if wd not in word_dict_check:
			if wd in wv_model:
				# print "---", wd
				word_dict_check[wd] = wd
				word_dict.append(wd)
				wd_count += 1

print wd_count

out_file = out_file = open("support_data/word_index.txt", 'a')

for ind in xrange(wd_count):
	print >> out_file, ind, word_dict[ind].encode('utf-8')
