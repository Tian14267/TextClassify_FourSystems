#coding: utf-8
import os
import jieba
import codecs
import numpy as np
from collections import Counter
import tensorflow.contrib.keras as kr

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


''' 制作vocab词汇表'''
##################################################################################
####  字级别   ############################################
def build_vocab(train_data,vocab_dir):
	labels = []
	contents = []
	with open(train_data,'r',encoding="utf-8") as f:
		all_line = f.readlines()
		for line in all_line:
			label,content = line.strip().split('\t')
			if content:
				labels.append(label)
				contents.append(content)
	f.close()
	all_data = []
	for content in contents:
		all_data.extend(content) ### 将所有汉字加入到numpy
	counter = Counter(all_data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	words, _ = list(zip(*count_pairs))
	words = ['<PAD>'] + list(words)
	#word_to_id = dict(zip(words, range(len(words))))  # 按字频排序后，{字：序号} 高频：0,1...
	write_file(vocab_dir, words)
	print("number of words:",len(words))
	return words

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

def make_word2voc(train_data,stop_word_dir,word2voc_file,word2vec_size):
	import logging
	import os
	from gensim.models import word2vec

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	lines = read_file(train_data)
	result = []
	for i, line in enumerate(lines):
		label, content = line.strip().split('\t')
		content_cut = jieba.cut(content, cut_all=False)
		content_cut_smooth = data_smooth(content_cut, stop_word_dir)  ## 数据平滑，只保留汉字
		one_line_content = []
		for x in content_cut_smooth:
			if x is not ' ':
				one_line_content.append(x)
		contents = ' '.join(one_line_content)
		result.append(contents)
	# result = ' '.join(contents)
	with open('./temp_file.txt', 'w', encoding="utf-8") as f2:
		for one_content in result:
			f2.write(str(one_content) + '\n')

	sentences = word2vec.LineSentence('./temp_file.txt')  ### 文件可以是一行，也可以是多行
	# sentences = contents
	model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=word2vec_size)
	model.wv.save_word2vec_format(word2voc_file, binary=False)
	os.remove('./temp_file.txt')

def create_voabulary_Jieba(train_data,word2vec,stop_word_dir,sequence_length):
	######  加载 word2vec ###################
	word2vec_list = read_file(word2vec)
	vocabulary_word2index = {}  ### word ：index
	#vocabulary_index2word = {}  ### index ：word
	len_vec = 0
	categories, cat_to_id = label_dict()  ### 加载label字典
	print("###### Load word2vec and dic_label Done! ########")
	for i, one_word2vec in enumerate(word2vec_list):
		if i ==0:
			pass
		else:
			word2vec_split = one_word2vec.split(' ')
			words = word2vec_split[0]
			vecs = np.array(word2vec_split[1:])
			vocabulary_word2index[words] = vecs
			#vocabulary_index2word[vec] = word
			len_vec = len(vecs)
	train_data_lines = read_file(train_data)
	total_data = []
	total_label = []
	print("###  Transform Data ...")
	for line in train_data_lines:
		label, content = line.strip().split('\t')
		label = cat_to_id[label]
		content_cut = jieba.cut(content, cut_all=False) ## 结巴分词
		content_cut_smooth = data_smooth(content_cut, stop_word_dir) ## 平滑
		vec_line = np.zeros([sequence_length,len_vec]) ## 设置一个空矩阵
		i = 0
		#print(len(content_cut_smooth))
		for word_content in content_cut_smooth:
			if word_content in vocabulary_word2index.keys():
				i += 1
				if i > sequence_length-1: ## 定长处理
					break
				vec_content = vocabulary_word2index[word_content]  ### 对一个词编码
				vec_content_np = np.array(vec_content, dtype=float)
				vec_line[i-1] = vec_content_np
		total_data.append(vec_line)
		total_label.append(label)
	Y = kr.utils.to_categorical(total_label, num_classes=len(cat_to_id)) ##3 one_hot处理
	X = np.array(total_data)
	print("####  Load and Transform data Done!")
	return X, Y

###################################
###################################
def make_files(train_data,word2vec,stop_word_dir,sequence_length,out_file):
	## 用来生成编码后的文本数据，存放于 out_file  ##########################################
	##  根据已经训练得到的 word2vec ，将训练数据的文本转换成向量编码，并保存在out_file中。
	######  加载 word2vec ##################################################################
	word2vec_list = read_file(word2vec)
	vocabulary_word2index = {}  ### word ：index
	len_vec = 0
	categories, cat_to_id = label_dict()  ### 加载label字典
	print("###### Load word2vec and dic_label Done! ########")
	for i, one_word2vec in enumerate(word2vec_list):
		if i == 0:
			pass
		else:
			word2vec_split = one_word2vec.split(' ')
			words = word2vec_split[0]
			vecs = np.array(word2vec_split[1:])
			vocabulary_word2index[words] = vecs
			# vocabulary_index2word[vec] = word
			len_vec = len(vecs)
	train_data_lines = read_file(train_data)
	print("###  Transform Data ...")
	with open(out_file, 'w', encoding='utf-8') as f:
		for line_num,line in enumerate(train_data_lines):
			label, content = line.strip().split('\t')
			label = cat_to_id[label]
			content_cut = jieba.cut(content, cut_all=False) ## 结巴分词
			content_cut_smooth =data_smooth(content_cut, stop_word_dir) ## 平滑
			vec_line = np.zeros([sequence_length,len_vec]) ## 设置一个空矩阵
			i = 0
			for word_content in content_cut_smooth:
				if word_content in vocabulary_word2index.keys():
					i += 1
					if i > sequence_length-1: ## 定长处理
						break
					vec_content = vocabulary_word2index[word_content]  ### 对一个词编码
					vec_content_np = np.array(vec_content, dtype=float)
					vec_line[i-1] = vec_content_np
			for line in vec_line:
				line_str = ''
				for one in range(len(line)-1):
					line_str = line_str +' '+str(line[one+1])
				f.write(str(line[0])+line_str+'\n')
			f.write(str(label)+ '\n')
			print("Num {:d} Finished ".format(line_num))
	f.close()


def ReadEncodeFile(file):
	###########################################
	## 读取被保存的编码后的训练数据
	###########################################
	file_list = read_file(file)
	text_all = []
	label_all = []

	text_num = int(len(file_list) / 201)
	for i in range(text_num):
		text_one = []
		for j in range(200): ## j : (0 ~ 199)
			one_line_file = file_list[i * 201 + j].split(' ')
			assert len(one_line_file) == 100
			text_line = np.array(one_line_file,dtype=float)
			text_one.append(text_line)
		assert len(file_list[(i+1)*201 - 1]) < 2
		label_one = file_list[(i + 1) * 201 - 1]

		text_tmp = np.array(text_one,dtype=float)
		text_all.append(text_tmp)
		label_all.append(label_one)

	X = np.array(text_all)
	categories, cat_to_id = label_dict()  ### 加载label字典
	Y = kr.utils.to_categorical(label_all, num_classes=len(cat_to_id))  ##3 one_hot处理

	return X,Y

##########################################################################################################
##########################################################################################################



'''制作label的序列'''
def label_dict():
	categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
	#categories = [x for x in categories]  #### 不是python3的话，需要对汉字编码
	cat_to_id = dict(zip(categories, range(len(categories))))
	return categories, cat_to_id

def create_voabulary(train_data,vocab_data,max_length):
	######  制作vocab 字典 ###################
	vocab_list = read_file(vocab_data)
	vocabulary_word2index = {}  ### word ：index
	vocabulary_index2word = {}  ### index ：word
	for i,vocab in enumerate(vocab_list):
		vocabulary_word2index[vocab] = i
		vocabulary_index2word[i] = vocab

	######## 文本编码  ##############################
	train_data_lines = read_file(train_data)
	categories, cat_to_id = label_dict()
	X = []
	Y = []
	for line in train_data_lines:
		#label,content = line.split('	')
		label,content = line.strip().split('\t')
		content = [vocabulary_word2index.get(e,0) for e in content]
		label = cat_to_id[label]
		X.append(content)
		Y.append(label)
	print("X[0]:",X[0])
	print("Y[0]:",Y[0])
	x_pad = kr.preprocessing.sequence.pad_sequences(X, max_length) #### 对数据进行定长处理
	y_pad = kr.utils.to_categorical(Y, num_classes=len(cat_to_id)) ### 对label进行onehot处理 ： 0 ==> [1,0,0,0,0,..,0,0]
	#out_data = (X,Y)
	return x_pad,y_pad

"""生成批次数据"""
def batch_iter(x, y, batch_size=64):
	data_len = len(x)
	num_batch = int((data_len - 1) / batch_size) + 1
	#print("Total batch:",num_batch)
	indices = np.random.permutation(np.arange(data_len)) ## 随机打乱原来的元素顺序  ## np.arange 生成的等差一维数组
	#print("Indices:",indices)
	x_shuffle = x[indices]
	y_shuffle = y[indices]

	for i in range(num_batch):
		start_id = i * batch_size
		end_id = min((i + 1) * batch_size, data_len)
		yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def batch_iter_Mix(x_char,x_word, y, batch_size=64):
	data_len = len(x_char)
	assert len(x_char) == len(x_word)
	#print(x_char.shape)
	#print(x_word.shape)
	#print(y.shape)
	num_batch = int((data_len - 1) / batch_size) + 1
	#print("Total batch:",num_batch)
	indices = np.random.permutation(np.arange(data_len)) ## 随机打乱原来的元素顺序  ## np.arange 生成的等差一维数组
	#print("Indices:",indices)
	x_char_shuffle = x_char[indices]
	x_word_shuffle = x_word[indices]
	y_shuffle = y[indices]

	for i in range(num_batch):
		start_id = i * batch_size
		end_id = min((i + 1) * batch_size, data_len)
		yield x_char_shuffle[start_id:end_id],x_word_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def compute_acc(pred,label): ### (Batch,10)
	batch,class_num = pred.shape
	num = 0
	for i in range(batch):
		if pred[i] == label[i]:
			num += 1
	acc = num / batch
	return acc

if __name__=="__main__":
	#build_vocab(train_data='../data/cnews.train.txt',vocab_dir='./cnews_vocab.txt') ### 制作词汇表
	#categories, cat_to_id=label_dict() ### 制作label词汇表
	#create_voabulary(train_data='../data/cnews.train.txt', vocab_data='./cnews_vocab.txt')

	indices = np.random.permutation(np.arange(50000))
	print("Stop!")
