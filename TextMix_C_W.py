#coding: utf-8
import tensorflow as tf
import config

def conv2d(input,filter_size,embedding_dim,input_channel, out_channel, padding="VALID",stride=1, data_format='NHWC', name=None):
	with tf.variable_scope(name):
		w = tf.get_variable("weight", [filter_size, embedding_dim, input_channel, out_channel],
							initializer=tf.truncated_normal_initializer(stddev=0.1))
		b = tf.get_variable("biase", [out_channel], initializer=tf.truncated_normal_initializer(0.0))
		conv = tf.nn.conv2d(input, w,strides=[1, stride, stride, 1], padding=padding, data_format=data_format)
		ret = tf.identity(tf.nn.bias_add(conv, b, data_format=data_format),name=name)
		conv = tf.contrib.layers.batch_norm(ret, is_training=True, scope='bn')
		relu = tf.nn.relu(features=conv, name='relu')

	return relu


####################  Char Model   ############################
class TextCNN_Char(object):
	def __init__(self,config,vocab_size,keep_prob):
		self.config = config
		self.is_training_flag = True
		self.keep_prob = keep_prob
		self.vocab_size = vocab_size

	def cnn_inference(self,input_x):
		#################  词嵌入: 对输入进行 embedding 嵌入  #########################################
		##### 对词汇表数据进行映射，词汇数 5000 ==> 128
		embedding = tf.get_variable('embedding_cnn_Char', [self.vocab_size, self.config.embedding_dim])
		embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)

		self.sentence_embeddings_expanded = tf.expand_dims(embedding_inputs, -1)  ### (?,200,128) ==> (?,200,128,1)
		#filter = [6,self.config.embedding_dim,1,128]

		conv1 = conv2d(input=self.sentence_embeddings_expanded, filter_size=6, embedding_dim=self.config.embedding_dim,
					   input_channel=1, out_channel=128, padding="VALID", stride=1, data_format='NHWC', name='conv1_Char')
		pool1 = tf.nn.max_pool(conv1, ksize=[1, self.config.seq_length - 6 + 1, 1, 1], strides=[1, 1, 1, 1],
								padding='VALID', name="pool1_Char")

		conv2 = conv2d(input=self.sentence_embeddings_expanded, filter_size=7, embedding_dim=self.config.embedding_dim,
					   input_channel=1, out_channel=128,padding="VALID", stride=1, data_format='NHWC', name='conv2_Char')
		pool2 = tf.nn.max_pool(conv2, ksize=[1, self.config.seq_length - 7 + 1, 1, 1], strides=[1, 1, 1, 1],
								 padding='VALID', name="pool2_Char")

		conv3 = conv2d(input=self.sentence_embeddings_expanded, filter_size=8, embedding_dim=self.config.embedding_dim,
					   input_channel=1, out_channel=128,padding="VALID", stride=1, data_format='NHWC', name='conv3_Char')
		pool3 = tf.nn.max_pool(conv3, ksize=[1, self.config.seq_length - 8 + 1, 1, 1], strides=[1, 1, 1, 1],
								 padding='VALID', name="pool3_Char")

		self.h = tf.concat([pool1,pool2,pool3], 3)  ### (?,1,1,384)
		self.h_flat = tf.reshape(self.h, [-1, 128*3])  ### (?,384)

		with tf.name_scope("dropout_cnn_Char"):
			self.h_drop = tf.nn.dropout(self.h_flat, keep_prob=self.keep_prob)

		self.h_dens = tf.layers.dense(self.h_drop, 128*3, activation=tf.nn.tanh, use_bias=True)
		with tf.name_scope("output_cnn_Char"):
			W_projection = tf.get_variable("W_projection_cnn_Char",shape=[128*3, self.config.num_classes],
										   initializer=tf.random_normal_initializer(stddev=0.1))
			b_projection =tf.get_variable("b_projection_cnn_Char",shape=[self.config.num_classes])
			logits = tf.matmul(self.h_dens,W_projection) + b_projection

		return logits

class TextRCNN_Char(object):
	def __init__(self,config,vocab_size,keep_prob):
		self.config = config
		self.vocab_size = vocab_size
		self.keep_prob = keep_prob

	def LSTM_cell(self):  ### 定义一层LSTM网络 + Dropout
		with tf.name_scope("LSTM_cell_Char"):
			lstm_cell  = tf.nn.rnn_cell.LSTMCell(self.config.hidden_dim, state_is_tuple=True)
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
		return lstm_cell

	def RCNN_inference(self,input_x):
		######### embedding X ####################
		#### input_x : [batch,seq_length]
		embedding = tf.get_variable('embedding_rcnn_Char',[self.vocab_size,self.config.embedding_dim])
		embedding_input = tf.nn.embedding_lookup(embedding,input_x)  ## [batch,seq_length,embedding_size] [16,200,128]

		#### 构建 LSTM 网络
		cells = [self.LSTM_cell() for i in range(2)]
		rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells,state_is_tuple=True)
		outputs,state = tf.nn.dynamic_rnn(cell=rnn_cell,inputs =embedding_input,dtype = tf.float32,scope="rnn_char")
		###  outputs : [batch, seq_length, hidden_size] [16,200,256]  state : [batch, hidden_size] [16,256]

		outputs_concat = tf.concat([outputs,embedding_input],axis=2) #[16,200,384]
		self.sentence_embeddings_expanded = tf.expand_dims(outputs_concat, -1)  #[16,200,384,1]
		#### pool_adv: [16,100,192,1]
		self.pool_adv = tf.nn.max_pool(self.sentence_embeddings_expanded, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],padding='VALID', name="pool_adv")
		self.out_cnn = self.CNN(self.pool_adv)  ### [16,1,1,384]
		#self.out_cnn = self.CNN_1(self.pool_adv)  ### [16,1,1,128]

		self.out_cnn_flat = tf.reshape(self.out_cnn, [-1, 128*3 ])  ### (?,384)

		with tf.name_scope("dropout_rcnn_Char"):
			self.h_drop = tf.nn.dropout(self.out_cnn_flat, keep_prob=self.keep_prob)

		self.h_dens = tf.layers.dense(self.h_drop, 128*3, activation=tf.nn.tanh, use_bias=True)
		with tf.name_scope("output_rcnn_Char"):
			W_projection = tf.get_variable("W_projection_rcnn_Char",shape=[128*3, self.config.num_classes],
										   initializer=tf.random_normal_initializer(stddev=0.1))
			b_projection =tf.get_variable("b_projection_rcnn_Char",shape=[self.config.num_classes])
			logits = tf.matmul(self.h_dens,W_projection) + b_projection

		return logits

	def CNN(self,input):
		############  input: [16,100,192,1]  #########################
		##   conv1: [16,95,1,128]  pool1：[16,1,1,128] -----------------↘
		##   conv2: [16,94,1,128]  pool1：[16,1,1,128] -------------- → concat: [16,1,1,384]  --> fc
		##   conv3: [16,93,1,128]  pool1：[16,1,1,128]------------------↗
		_, size_w,size_h,_= input.get_shape().as_list()
		conv1 = conv2d(input=input, filter_size=6, embedding_dim=size_h,input_channel=1, out_channel=128,
					   padding="VALID", stride=1, data_format='NHWC', name='conv1_rcnn_Char')
		pool1 = tf.nn.max_pool(conv1, ksize=[1, size_w - 6 + 1, 1, 1], strides=[1, 1, 1, 1],padding='VALID', name="pool1_rcnn_Char")

		conv2 = conv2d(input=input, filter_size=7, embedding_dim=size_h, input_channel=1, out_channel=128,
					   padding="VALID", stride=1, data_format='NHWC', name='conv2_rcnn_Char')
		pool2 = tf.nn.max_pool(conv2, ksize=[1, size_w - 7 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool2_rcnn_Char")

		conv3 = conv2d(input=input, filter_size=8, embedding_dim=size_h, input_channel=1, out_channel=128,
					   padding="VALID", stride=1, data_format='NHWC', name='conv3_rcnn_Char')
		pool3 = tf.nn.max_pool(conv3, ksize=[1, size_w - 8 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool3_rcnn_Char")

		out_h = tf.concat([pool1, pool2, pool3], 3)  ### (?,1,1,384)

		return out_h

####################  Word Model   ###########################
class Text_word_CNN(object):
	def __init__(self,config,keep_prob):
		self.config = config
		self.keep_prob = keep_prob

	def cnn(self,input_x):
		### input_x: [batch,seq_len,word2vec_size]
		self.input_expanded = tf.expand_dims(input_x, -1)  ### (?,200,100) ==> (?,200,100,1)

		conv1 = conv2d(input=self.input_expanded, filter_size=6, embedding_dim=self.config.word2vec_size,
					   input_channel=1, out_channel=128, padding="VALID", stride=1, data_format='NHWC', name='conv1_word')
		pool1 = tf.nn.max_pool(conv1, ksize=[1, self.config.seq_length - 6 + 1, 1, 1], strides=[1, 1, 1, 1],
							   padding='VALID', name="pool1_word")

		conv2 = conv2d(input=self.input_expanded, filter_size=7, embedding_dim=self.config.word2vec_size,
					   input_channel=1, out_channel=128, padding="VALID", stride=1, data_format='NHWC', name='conv2_word')
		pool2 = tf.nn.max_pool(conv2, ksize=[1, self.config.seq_length - 7 + 1, 1, 1], strides=[1, 1, 1, 1],
							   padding='VALID', name="pool2_word")

		conv3 = conv2d(input=self.input_expanded, filter_size=8, embedding_dim=self.config.word2vec_size,
					   input_channel=1, out_channel=128, padding="VALID", stride=1, data_format='NHWC', name='conv3_word')
		pool3 = tf.nn.max_pool(conv3, ksize=[1, self.config.seq_length - 8 + 1, 1, 1], strides=[1, 1, 1, 1],
							   padding='VALID', name="pool3_word")

		self.h = tf.concat([pool1, pool2, pool3], 3)  ### (?,1,1,384)
		self.h_flat = tf.reshape(self.h, [-1, 128 * 3])  ### (?,384)

		with tf.name_scope("dropout_word"):
			self.h_drop = tf.nn.dropout(self.h_flat, keep_prob=self.keep_prob)

		self.h_dens = tf.layers.dense(self.h_drop, 128*3, activation=tf.nn.tanh, use_bias=True)
		with tf.name_scope("output_word"):
			W_projection = tf.get_variable("W_projection_word",shape=[128*3, self.config.num_classes],
										   initializer=tf.random_normal_initializer(stddev=0.1))
			b_projection =tf.get_variable("b_projection_word",shape=[self.config.num_classes])
			logits = tf.matmul(self.h_dens,W_projection) + b_projection

		return logits

class TextRCNN_word(object):
	def __init__(self,config,keep_prob):
		self.config = config
		self.keep_prob = keep_prob

	def LSTM_cell(self):  ### 定义一层LSTM网络 + Dropout
		with tf.name_scope("LSTM_cell_word"):
			lstm_cell  = tf.nn.rnn_cell.LSTMCell(self.config.hidden_dim, state_is_tuple=True)
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
		return lstm_cell

	def RCNN_inference(self,input_x):
		######### embedding X ####################
		#### input_x : [batch,seq_length,word2vec_size]

		#### 构建 LSTM 网络
		cells = [self.LSTM_cell() for i in range(2)]
		rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells,state_is_tuple=True)
		outputs,state = tf.nn.dynamic_rnn(cell=rnn_cell,inputs =input_x,dtype = tf.float32,scope="rnn_word")
		###  outputs : [batch, seq_length, hidden_size] [16,200,256]  state : [batch, hidden_size] [16,256]

		outputs_concat = tf.concat([outputs,input_x],axis=2) #[16,200,384]
		self.sentence_embeddings_expanded = tf.expand_dims(outputs_concat, -1)  #[16,200,384,1]
		#### pool_adv: [16,100,192,1]
		self.pool_adv = tf.nn.max_pool(self.sentence_embeddings_expanded, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],padding='VALID', name="pool_adv")
		self.out_cnn = self.CNN(self.pool_adv)  ### [16,1,1,384]
		#self.out_cnn = self.CNN_1(self.pool_adv)  ### [16,1,1,128]

		self.out_cnn_flat = tf.reshape(self.out_cnn, [-1, 128*3 ])  ### (?,384)

		with tf.name_scope("dropout_rcnn_word"):
			self.h_drop = tf.nn.dropout(self.out_cnn_flat, keep_prob=self.keep_prob)

		self.h_dens = tf.layers.dense(self.h_drop, 128*3, activation=tf.nn.tanh, use_bias=True)
		with tf.name_scope("output_rcnn_word"):
			W_projection = tf.get_variable("W_projection_rcnn_word",shape=[128*3, self.config.num_classes],
										   initializer=tf.random_normal_initializer(stddev=0.1))
			b_projection =tf.get_variable("b_projection_rcnn_word",shape=[self.config.num_classes])
			logits = tf.matmul(self.h_dens,W_projection) + b_projection

		return logits

	def CNN_1(self,input):
		_, size_w,size_h,_= input.get_shape().as_list()
		conv1 = conv2d(input=input, filter_size=6, embedding_dim=size_h,input_channel=1, out_channel=128,
					   padding="VALID", stride=1, data_format='NHWC', name='conv1_rcnn_word')
		pool1 = tf.nn.max_pool(conv1, ksize=[1, size_w - 6 + 1, 1, 1], strides=[1, 1, 1, 1],padding='VALID', name="pool1_rcnn_word")
		return pool1

	def CNN(self,input):
		############  input: [16,100,192,1]  #########################
		##   conv1: [16,95,1,128]  pool1：[16,1,1,128] -----------------↘
		##   conv2: [16,94,1,128]  pool1：[16,1,1,128] -------------- → concat: [16,1,1,384]  --> fc
		##   conv3: [16,93,1,128]  pool1：[16,1,1,128]------------------↗
		_, size_w,size_h,_= input.get_shape().as_list()
		conv1 = conv2d(input=input, filter_size=6, embedding_dim=size_h,input_channel=1, out_channel=128,
					   padding="VALID", stride=1, data_format='NHWC', name='conv1_rcnn_word')
		pool1 = tf.nn.max_pool(conv1, ksize=[1, size_w - 6 + 1, 1, 1], strides=[1, 1, 1, 1],padding='VALID', name="pool1_rcnn_word")

		conv2 = conv2d(input=input, filter_size=7, embedding_dim=size_h, input_channel=1, out_channel=128,
					   padding="VALID", stride=1, data_format='NHWC', name='conv2_rcnn_word')
		pool2 = tf.nn.max_pool(conv2, ksize=[1, size_w - 7 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool2_rcnn_word")

		conv3 = conv2d(input=input, filter_size=8, embedding_dim=size_h, input_channel=1, out_channel=128,
					   padding="VALID", stride=1, data_format='NHWC', name='conv3_rcnn_word')
		pool3 = tf.nn.max_pool(conv3, ksize=[1, size_w - 8 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool3_rcnn_word")

		out_h = tf.concat([pool1, pool2, pool3], 3)  ### (?,1,1,384)

		return out_h

class Mix_model(object):
	def __init__(self,vocab_size):
		self.textcnn_char = TextCNN_Char(config, vocab_size, keep_prob=config.dropout_keep_prob)
		self.textrcnn_char = TextRCNN_Char(config, vocab_size, keep_prob=config.dropout_keep_prob)

		self.textrcnn_word = TextRCNN_word(config, keep_prob=config.dropout_keep_prob)
		self.textcnn_word = Text_word_CNN(config, keep_prob=config.dropout_keep_prob)

	def __call__(self, input_char,input_word):
		result = self.forward(input_char,input_word)
		return result

	def forward(self,input_char,input_word):
		logits_cnn = self.textcnn_char.cnn_inference(input_char)  ### (?,10)
		logits_rcnn = self.textrcnn_char.RCNN_inference(input_char)  ### (?,10)

		logits_RCNN = self.textrcnn_word.RCNN_inference(input_word)  ### (?,10)
		logits_CNN = self.textcnn_word.cnn(input_word)  ### (?,10)

		logits = tf.concat([logits_cnn,logits_rcnn,logits_RCNN,logits_CNN],axis=1)  ### (?,40)

		'''
		with tf.name_scope("Mix_output"):
			W_mix = tf.get_variable("W_projection_Mix",shape=[40, config.num_classes],
										   initializer=tf.random_normal_initializer(stddev=0.1))
			b_mix =tf.get_variable("b_projection__Mix",shape=[config.num_classes])
			logits_out = tf.matmul(logits,W_mix) + b_mix
		'''
		with tf.name_scope("Mix_output"):
			W_mix = tf.get_variable("W_projection_Mix",shape=[40, config.num_classes],
										   initializer=tf.random_normal_initializer(stddev=0.1))
			V_mix = tf.get_variable("V_projection_Mix", shape=[config.num_classes, config.num_classes],
									initializer=tf.random_normal_initializer(stddev=0.1))

			out_1 = tf.tanh(tf.matmul(logits,W_mix)) ## (batch, 10)
			logits_out = tf.matmul(out_1,V_mix,name="Mix")

		return logits_out





def Loss(logits, label):
	with tf.name_scope("loss"):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
		loss = tf.reduce_mean(cross_entropy)
	return loss

def Acc(logits, labels):  ### 计算 acc 正确率
	##########  输入 logits 和 label都是one_hot编码
	####  labels = [[1,0,0,0,0,...,0],[0,1,0,0,...,0],...]
	####  logits =
	with tf.name_scope("acc"):
		pred = tf.argmax(tf.nn.softmax(logits), 1)  #### 计算预测值 （64，）
		correct_pred = tf.equal(tf.argmax(labels, 1), pred)
		acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	return acc


if __name__=="__main__":
	import config
	#input = tf.zeros([4, 200,128], dtype=tf.float32)
	#input = tf.placeholder(tf.int32, [4, 200, 128])
	input_x_char = tf.placeholder(tf.int32, [4, config.seq_length], name='input_x_c')
	input_x_word = tf.placeholder(tf.float32, [4, config.seq_length, config.word2vec_size], name='input_x_w')
	input_y = tf.placeholder(tf.float32, [4, config.num_classes], name='input_y')

	'''
	#########  Char Model  ####################################################
	textcnn = TextCNN_Char(config, 5000, keep_prob=config.dropout_keep_prob)
	textrcnn = TextRCNN_Char(config, 5000, keep_prob=config.dropout_keep_prob)
	logits_cnn = textcnn.cnn_inference(input_x_char)  ### (?,10)
	logits_rcnn = textrcnn.RCNN_inference(input_x_char)  ### (?,10)

	########  Word Model  ########################################################
	textrcnn = TextRCNN_word(config, keep_prob=config.dropout_keep_prob)
	logits_RCNN = textrcnn.RCNN_inference(input_x_word)  ### (?,10)
	textcnn = Text_word_CNN(config, keep_prob=config.dropout_keep_prob)
	logits_CNN = textcnn.cnn(input_x_word)  ### (?,10)
	'''
	mixmodel = Mix_model(vocab_size=5000)
	rel = mixmodel(input_x_char,input_x_word)

	#logits = logits_cnn + logits_rcnn + logits_RCNN + logits_CNN
	print('over!')
