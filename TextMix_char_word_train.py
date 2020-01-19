#coding: utf-8
import numpy as np
import tensorflow as tf
import tools
import jieba
import os
import time
import logging
import TextMix_C_W_model as TM
import tensorflow.contrib.keras as kr
import config
os.environ['CUDA_VISIBLE_DEVICES']='1'

flages=tf.flags
FLAGS = flages.FLAGS
flages.DEFINE_float("learning_rate",0.001,"learning rate")
flages.DEFINE_string("stop_word","./stop_word.txt","stop word data path")
flages.DEFINE_string("word2vec","./word2voc_all.bin","word2voc data path")
flages.DEFINE_boolean("model_restore",False,"wheater to restore model")

flages.DEFINE_string("vocab_dir","../data/cnews_vocab.txt","vocab data path")
flages.DEFINE_string("train_data","../data/cnews.train.txt","training data path")
flages.DEFINE_string("val_data","../data/cnews.val.txt","val data path")
flages.DEFINE_string("test_data","../data/cnews.test.txt","test data path")


def preprocess_data(vocab_data,word2vec,stop_word_dir,train_data,sequence_length):
	######  制作vocab 字典 ###################
	vocab_list = tools.read_file(vocab_data)
	vocabulary_word2index_char = {}  ### word ：index
	for i, vocab in enumerate(vocab_list):
		vocabulary_word2index_char[vocab] = i

	######  加载 word2vec ###################
	word2vec_list = tools.read_file(word2vec)
	vocabulary_word2index_word = {}
	len_vec = 0
	for i, one_word2vec in enumerate(word2vec_list):
		if i ==0:
			pass
		else:
			word2vec_split = one_word2vec.split(' ')
			words = word2vec_split[0]
			vecs = np.array(word2vec_split[1:])
			vocabulary_word2index_word[words] = vecs
			len_vec = len(vecs)

	######## 文本编码  ##############################
	train_data_lines = tools.read_file(train_data)
	categories, cat_to_id = tools.label_dict()
	X_char = []
	Y = []
	total_data = []
	for line in train_data_lines:
		label, content = line.strip().split('\t')
		contents = [vocabulary_word2index_char.get(e, 0) for e in content]
		label = cat_to_id[label]
		X_char.append(contents)
		Y.append(label)

		content_cut = jieba.cut(content, cut_all=False)  ## 结巴分词
		content_cut_smooth = tools.data_smooth(content_cut, stop_word_dir)  ## 平滑
		vec_line = np.zeros([sequence_length, len_vec])  ## 设置一个空矩阵
		i = 0
		# print(len(content_cut_smooth))
		for word_content in content_cut_smooth:
			if word_content in vocabulary_word2index_word.keys():
				i += 1
				if i > sequence_length - 1:  ## 定长处理
					break
				vec_content = vocabulary_word2index_word[word_content]  ### 对一个词编码
				vec_content_np = np.array(vec_content, dtype=float)
				vec_line[i - 1] = vec_content_np
		total_data.append(vec_line)
	X = kr.preprocessing.sequence.pad_sequences(X_char, sequence_length)
	X_word = np.array(total_data)
	Y_all = kr.utils.to_categorical(Y, num_classes=len(cat_to_id))

	return X,X_word,Y_all

def TextMix_train():
	###########  load data  ###################
	if not os.path.exists(FLAGS.vocab_dir):
		words = tools.build_vocab(train_data=FLAGS.train_data, vocab_dir=FLAGS.vocab_dir)  ### 制作词汇表
	else:
		words = tools.read_file(FLAGS.vocab_dir)
	vocab_size = len(words)

	train_char_X, train_word_X,train_Y = preprocess_data(vocab_data=FLAGS.vocab_dir,word2vec=FLAGS.word2vec,
					stop_word_dir=FLAGS.stop_word,train_data=FLAGS.train_data,sequence_length=config.seq_length)

	val_char_X, val_word_X, val_Y = preprocess_data(vocab_data=FLAGS.vocab_dir, word2vec=FLAGS.word2vec,
														  stop_word_dir=FLAGS.stop_word, train_data=FLAGS.val_data,
														  sequence_length=config.seq_length)

	print("Data deal down!")
	###########################################

	input_x_char = tf.placeholder(tf.int32, [None, config.seq_length], name='input_x')
	input_x_word = tf.placeholder(tf.float32, [None, config.seq_length, config.word2vec_size], name='input_x')
	input_y = tf.placeholder(tf.float32, [None, config.num_classes], name='input_y')

	#########  Char Model  ####################################################
	textcnn = TM.TextCNN_Char(config, vocab_size, keep_prob=config.dropout_keep_prob)
	textrcnn = TM.TextRCNN_Char(config, vocab_size, keep_prob=config.dropout_keep_prob)
	logits_cnn = textcnn.cnn_inference(input_x_char)  ### (?,10)
	logits_rcnn = textrcnn.RCNN_inference(input_x_char)  ### (?,10)

	########  Word Model  ########################################################
	textrcnn = TM.TextRCNN_word(config, keep_prob=config.dropout_keep_prob)
	logits_RCNN = textrcnn.RCNN_inference(input_x_word)  ### (?,10)
	textcnn = TM.Text_word_CNN(config, keep_prob=config.dropout_keep_prob)
	logits_CNN = textcnn.cnn(input_x_word)  ### (?,10)

	###### 模型融合
	logits = logits_cnn + logits_rcnn + logits_RCNN + logits_CNN

	############# 计算 loss 和 acc ######################################
	loss = TM.Loss(logits=logits, label=input_y)
	acc = TM.Acc(logits=logits, labels=input_y)

	global_step = tf.Variable(0, name='global_step', trainable=False)
	#learning_rate = FLAGS.learning_rate
	learning_rate = tf.train.exponential_decay(
		learning_rate=FLAGS.learning_rate,
		global_step=global_step,
		decay_steps=2500,
		decay_rate=0.5,
		staircase=True)

	optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss,global_step=global_step)
	#optim = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(loss=loss,global_step=global_step)

	tensorboard_dir = 'tensorboard/textcnn_word'
	tf.summary.scalar("loss", loss)
	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter(tensorboard_dir)

	saver = tf.train.Saver(max_to_keep=3) ### 保存模型
	model_save_dir = 'checkpoints/'
	train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
	model_name = 'TextMixNet_{:s}.ckpt'.format(str(train_start_time))
	model_save_path = os.path.join(model_save_dir, model_name)

	logging.basicConfig(filename='./checkpoints/' + model_name + '.log',
						format='%(asctime)s - %(pathname)s - %(levelname)s: %(message)s',
						level=logging.DEBUG, filemode='a', datefmt='%Y-%m-%d%I:%M:%S %p')

	sess_config = tf.ConfigProto(allow_soft_placement=True)
	sess_config.gpu_options.allow_growth = True
	sess = tf.Session(config=sess_config)

	with sess.as_default():
		if FLAGS.model_restore:
			weights_path = './checkpoints/TextWordRCNNnet_2019-11-06-16-07-55.ckpt-12500'
			saver.restore(sess=sess, save_path=weights_path)
			step = sess.run(tf.train.get_global_step())
			writer.add_graph(sess.graph)
			infor = '##### Restore model : ' + weights_path + '  ########'
			logging.info(infor)
			print(infor)
		else:
			step = 0
			init = tf.global_variables_initializer()
			sess.run(init)
			writer.add_graph(sess.graph)
		print('First step is:', step)
		num_batch = int((len(train_char_X) - 1) / config.batch_size) + 1  ### 总batch数
		acc_begain = 0
		for epoch in range(config.epochs):
			batch_train = tools.batch_iter_Mix(train_char_X, train_word_X,train_Y, config.batch_size)  ### 生成批次数据
			Begain_learn_rate = FLAGS.learning_rate
			for x_char_batch,x_word_batch, y_batch in batch_train:
				step += 1
				_, learn_rate, train_loss_value, train_pred, train_acc, merge_summary_value = sess.run(
					[optim, learning_rate, loss, logits, acc, merged_summary],
					feed_dict={input_x_char: x_char_batch,input_x_word:x_word_batch, input_y: y_batch})
				############# 输出 learning_rate
				if Begain_learn_rate != learn_rate:
					information = '############ New Learning_Rate {:6f} in step {:d}  ###########'.format(learn_rate,
																										  step)
					logging.info(information)
					print(information)
					Begain_learn_rate = learn_rate
				if step % 10 == 0:
					information = '## Epoch {:d} Step_Train / Total_Batch: {:d} / {:d}  train_loss= {:5f} train_acc={:5f}'. \
						format(epoch, step, num_batch, train_loss_value, train_acc)
					print(information)  ### 输出到屏幕
					logging.info(information)  ### 输出到log文件

				if step % 500 == 0:  ### 每 500 步进行一次验证，并保存最优模型
					val_acc_all = 0
					val_loss_all = 0
					val_step = 0
					batch_val = tools.batch_iter_Mix(val_char_X, val_word_X, val_Y, config.batch_size)  ### 生成批次数据
					for x_char_val,x_word_val, y_val in batch_val:
						if x_char_val.shape[0] < config.batch_size:
							pass
						else:
							_, val_loss_value, val_pred, val_acc, merge_summary_value = sess.run(
								[optim, loss, logits, acc, merged_summary], feed_dict={input_x_char: x_char_val,
													input_x_word:x_word_val,input_y: y_val})
							writer.add_summary(merge_summary_value, step)
							val_acc_all = val_acc_all + val_acc
							val_loss_all = val_loss_all + val_loss_value
							val_step += 1
					ave_acc = val_acc_all / val_step
					ave_loss = val_loss_all / val_step
					information = "############## Val_loss = {:5f}   Val_acc = {:5f}   #####################". \
						format(ave_loss, ave_acc)
					print(information)  ### 输出到屏幕
					logging.info(information)  ### 输出到log文件

					if (ave_acc - acc_begain) > 0.0005:
						saver.save(sess, model_save_path, global_step=step)
						acc_begain = ave_acc


def TextMix_test():
	###########  load data  ###################
	print("############## Begain to load data!  ############")
	if not os.path.exists(FLAGS.vocab_dir):
		words = tools.build_vocab(train_data=FLAGS.train_data, vocab_dir=FLAGS.vocab_dir)  ### 制作词汇表
	else:
		words = tools.read_file(FLAGS.vocab_dir)
	vocab_size = len(words)

	test_char_X, test_word_X,test_Y = preprocess_data(vocab_data=FLAGS.vocab_dir,word2vec=FLAGS.word2vec,
					stop_word_dir=FLAGS.stop_word,train_data=FLAGS.test_data,sequence_length=config.seq_length)

	print("Load data down!")

	input_x_char = tf.placeholder(tf.int32, [None, config.seq_length], name='input_x')
	input_x_word = tf.placeholder(tf.float32, [None, config.seq_length, config.word2vec_size], name='input_x')
	input_y = tf.placeholder(tf.float32, [None, config.num_classes], name='input_y')

	model_path = 'checkpoints/TextMixNet_2019-11-07-16-53-09.ckpt-35500'

	sess_config = tf.ConfigProto(allow_soft_placement=True)
	sess_config.gpu_options.allow_growth = True
	sess = tf.Session(config=sess_config)

	#########  Char Model  ####################################################
	textcnn = TM.TextCNN_Char(config, vocab_size, keep_prob=config.dropout_keep_prob)
	textrcnn = TM.TextRCNN_Char(config, vocab_size, keep_prob=config.dropout_keep_prob)
	logits_cnn = textcnn.cnn_inference(input_x_char)  ### (?,10)
	logits_rcnn = textrcnn.RCNN_inference(input_x_char)  ### (?,10)

	########  Word Model  ########################################################
	textrcnn = TM.TextRCNN_word(config, keep_prob=config.dropout_keep_prob)
	logits_RCNN = textrcnn.RCNN_inference(input_x_word)  ### (?,10)
	textcnn = TM.Text_word_CNN(config, keep_prob=config.dropout_keep_prob)
	logits_CNN = textcnn.cnn(input_x_word)  ### (?,10)

	###### 模型融合
	logits = logits_cnn + logits_rcnn + logits_RCNN + logits_CNN
	####################
	
	loss = TM.Loss(logits=logits, label=input_y)
	acc = TM.Acc(logits=logits, labels=input_y)


	saver = tf.train.Saver()
	saver.restore(sess=sess, save_path=model_path)

	batch_test = tools.batch_iter_Mix(test_char_X, test_word_X,test_Y, config.batch_size)  ### 生成批次数据
	i = 0
	all_acc = 0
	for  x_char_test,x_word_test, y_test in batch_test:
		i += 1
		test_loss,test_acc = sess.run([loss,acc],feed_dict = {input_x_char: x_char_test,
													input_x_word:x_word_test,input_y: y_test })
		all_acc = all_acc + test_acc

	print("Average acc : ",(all_acc/i))

if __name__=="__main__":
	#TextMix_train()
	TextMix_test()



