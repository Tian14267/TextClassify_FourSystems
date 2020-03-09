#coding: utf-8
"""
@author: FrankFan
@time:19-12-17

"""
import os
import jieba
import codecs
import random

def covert_file(filename,outfile="covert.txt"):
	try:
		with codecs.open(filename, 'r', "utf-8") as f_in:
			new_content = f_in.read()
			f_out = codecs.open(outfile, 'w', "UTF-8")
			f_out.write(new_content)
		f_out.close()
	except IOError as error:
		print("I/O error: {0}".format(error))

def write_file(file,data): ### 输入数组
	with open(file,'w',encoding='utf-8') as f:
		f.write('\n'.join(data) + '\n')
	f.close()

def read_file(file):
	try:
		with open(file,'r',encoding='utf-8') as f:
			lines = f.readlines()
			new_lines = []
			for line in lines:
				new_line = line.strip().replace('\n','')
				new_lines.append(new_line)
		f.close()
	except:
		covert_file(file, outfile="covert.txt")
		with open("covert.txt",'r',encoding='utf-8') as f:
			lines = f.readlines()
			new_lines = []
			for line in lines:
				new_line = line.strip().replace('\n','')
				new_lines.append(new_line)
		f.close()
		os.remove("covert.txt")
	return new_lines


#### 词汇级别   #############################################
def jieba_data(train_data,stop_word_file,vocab_dir):
	#### train_data: 训练数据
	#### stop_word_file：停用词列表
	#### vocab_dir：词典输出列表

	contents = []
	all_lines = read_file(train_data)
	for i,line in enumerate(all_lines):
		label, content = line.strip().split('\t')
		content_cut = jieba.cut(content, cut_all=False)
		for content_one in content_cut:
			if content_one not in contents:
				contents.append(content_one)
		print("line {:d} is ok !".format(i))
	print("contents len: ",len(contents))

	all_stop_words = read_file(stop_word_file)
	new_contents = []
	for content in contents:
		if content in all_stop_words:
			pass
		else:
			new_contents.append(content)
	print("New contents len: ",len(new_contents))
	write_file(vocab_dir, new_contents)

	return new_contents

def data_smooth(data,stop_word_dir):
	###  1: 去除数组中的非中文字符。包括标点符号，英文字母，特殊符号等
	###  2: 去除停用词
	###  3：去除单字
	### data: 一个一维数组:['去除','中的','括标',...]
	import re
	zhmodel = re.compile(u'[\u4e00-\u9fa5]')  # 检查中文
	all_stop_words = read_file(stop_word_dir)  ## 加载停用词
	new_data = []
	for one_data in data:
		match = zhmodel.search(one_data)
		if match and one_data not in all_stop_words: ###去除非中文和停用词
			if len(one_data) >1: ### 去除单字
				new_data.append(one_data)
	return new_data

def smooth_sentence(sentence):
	#### 去除句子中的非中文单字
	import re
	zhmodel = re.compile(u'[\u4e00-\u9fa5]')  # 检查中文
	out_data = ""
	for data in sentence:
		match = zhmodel.search(data)
		if match:
			out_data = out_data+data
	return out_data

def read_news(news_path):
	files_name = os.listdir(news_path)  # 采用listdir来读取所有文件
	file_full_name = [os.path.join(news_path,file) for file in files_name]
	file_contents = []
	i=0
	file_failed = [] ### 统计读取失败的文件
	for file in file_full_name:
		try:
			with open(file, 'r', encoding='utf-8') as f:
				lines = f.readlines()
				for line in lines:
					new_line = line.strip().replace('\n', '')
					file_contents.append(new_line)
			f.close()
			i = i+1
		except:
			file_failed.append(file)
		print("Deal file num: ",i)
	print("读取失败的文件：",file_failed)
	random.shuffle(file_contents) ### 随机打乱
	return file_contents

def make_word2voc(data_lines,stop_word_dir,word2voc_file,word2vec_size):
	import logging
	import os
	from gensim.models import word2vec

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	result = []
	for i, line in enumerate(data_lines):
		content = line.strip().replace('\t','')
		content_cut = jieba.cut(content, cut_all=False)
		content_cut_smooth = data_smooth(content_cut, stop_word_dir)  ## 数据平滑，只保留汉字
		one_line_content = []
		for x in content_cut_smooth:
			if x is not ' ':
				one_line_content.append(x)
		contents = ' '.join(one_line_content)
		result.append(contents)
	with open('./temp_file.txt', 'w', encoding="utf-8") as f2:
		for one_content in result:
			f2.write(str(one_content) + '\n')
	sentences = word2vec.LineSentence('./temp_file.txt')  ### 文件可以是一行，也可以是多行
	# sentences = contents
	model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=5, size=word2vec_size)
	model.wv.save_word2vec_format(word2voc_file, binary=True)
	os.remove('./temp_file.txt')


if __name__=="__main__":
	new_path = "/home/hj/smbshare/fffan/Data/Sohu_News/"
	new_lines = read_news(new_path)
	print("All Len: ",len(new_lines))
	cutnum = int(len(new_lines) / 4)
	lines = new_lines[0:cutnum]
	new_lines = []
	print("Select: ",len(lines))
	make_word2voc(lines,stop_word_dir="./stop_word.txt",word2voc_file="./word2vec.model",word2vec_size=200)

	print("Done!")
