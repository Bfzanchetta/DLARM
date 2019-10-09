%matplotlib inline
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import pylab

def one_hot(ind,vocab_size):
	rec = np.zeros(vocab_size)
	rec[ind] = 1
	return rec

def create_training_data(corpus_raw,WINDOW_SIZE = 2):
	words_list = []
	
	for sent in corpus_raw.split('.'):
		for w in sent.split():
			if w != '.':
				words_list.append(w.split('.')[0])
				##Aqui não fez muito sentido, mas deve estar splitando as palavras
	words_list = set(words_list)
	##o que set(array) faz? don't know
	word2ind = {}
	
	ind2word = {}
	##cria dois dicionários auxiliares
	vocab_size = len(words_list)
	#var aux para for's
	for i,w in enumerate(words_list): ##o que faz esse enumerate? O Jobbe acha que indexa cada elemento como se virasse um objeto com mais uma atributo
		word2ind[w] = i
		ind2word[i] = w
	##Isso aqui é importantíssimo
	print word2ind
	sentences_list = corpus_raw.split('.')
	sentences = []
	
	for sent in sentences_list:
		sent_array = sent.split()
		sent_array = [s.split('.')[0] for s in sent_array]
		sentences.append(sent_array)
		
	data_recs = []
	
	for sent in sentences:
		for ind,w in enumerate(sent):
			rec = []
			for nb_w in sent[max(ind - WINDOW_SIZE, 0) : min(ind + WINDOW_SIZE,len(sent)) +1] :
				if nb_w != w:
					rec.append(nb_w)
				data_recs.append([rec,w])
				
	x_train,y_train = [],[]
	
	for rec in data_recs:
		input_ = np.zeros(vocab_size)
		for i in xrange(WINDOW_SIZE-1):
			input_ += one_hot(word2ind[ rec[0][i] ], vocab_size)
		input_ = input_/len(rec[0])
		x_train.append(input_)
		y_train.append(one_hot(word2ind[ rec[1] ],vocab_size))
		
	return x_train, y_train, word2ind, ind2word, vocab_size

corpus_raw = "Breno foi formado em Engenharia na Universidade Federal de Rio Grande FURG em 2017 e atualmente integra o grupo de estudantes de mestrado do grupo GPPD"
	
corpus_raw = (corpus_raw).lower()
x_train, y_train, word2ind, ind2word, vocab_size = create_training_data(corpus_raw,2)

emb_dims = 128
learning_rate = 0.001

x = tf.placeholder(tf.float32,[None,vocab_size])
y = tf.placeholder(tf.float32,[None,vocab_size])

W = tf.Variable(tf.random_normal([vocab_size,emb_dims],mean=0.0,stddev=0.02,dtype=tf.float32))
b = tf.Variable(tf.random_normal([emb_dims],mean=0.0,stddev=0.02,dtype=tf.float32))
W_outer = tf.Variable(tf.random_normal([emb_dims,vocab_size],mean=0.0,stddev=0.02,dtype=tf.float32))
b_outer = tf.Variable(tf.random_normal([vocab_size],mean=0.0,stddev=0.02,dtype=tf.float32))

hidden = tf.add(tf.matmul(x,W),b)
logits = tf.add(tf.matmul(hidden,W_outer),b_outer)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

epochs,batch_size = 100,10
batch = len(x_train)//batch_size

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print 'was here'
	for epoch in xrange(epochs):
		batch_index = 0
		for batch_num in xrange(batch):
			x_batch = x_train[batch_index: batch_index + batch_size]
			y_batch = y_train[batch_index: batch_index + batch_size]
			sess.run(optimizer,feed_dict={x: x_batch,y: y_batch})
			print('epoch:',epoch,'loss :', sess.run(cost,feed_dict={x: x_batch,y: y_batch}))
	W_embed_trained = sess.run(W)
	
W_embedded = TSNE(n_components=2).fit_transform(W_embed_trained)
plt.figure(figsize=(10,10))
for i in xrange(len(W_embedded)):
	plt.text(W_embedded[i,0],W_embedded[i,1],ind2word[i])
	
plt.xlim(-150,150)
plt.ylim(-150,150)
		
		