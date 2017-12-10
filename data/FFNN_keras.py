from __future__ import division
from data_utils import load_task
import csv
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils import np_utils


W2VEC_LEN = 50
GLOVE_path = "./raw/glove.6B/glove.6B.50d.txt"
_W2VEC = None
miss = 0  #Total words which we cudnt find in GloVe

model = Sequential()
def nn(X_train, y_train, X_test, y_test):
	print X_train.shape
	print y_train.shape
	print X_test.shape
	print y_test.shape

	batch_size = 32
	nb_epoch = 50

	model.add(Dense(200, input_dim=2*W2VEC_LEN, activation='relu'))
	model.add(Dense(200, activation='relu'))
	model.add(Dense(2))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, validation_data=(X_test, y_test)

def fetch_data(path="./raw/", task_id = 6):
	trn_data, tst_data = load_task(path, task_id)
	return trn_data, tst_data

def vectorize(sentence):
	n_words = len(sentence)
	sent_vec = np.zeros((n_words, W2VEC_LEN))
	if _W2VEC is None:
		global _W2VEC
		reader = csv.reader(open(GLOVE_path), delimiter=' ', quoting=csv.QUOTE_NONE) 
		_W2VEC = {line[0]: np.array(list(map(float, line[1: ]))) for line in reader}

	for idx, word in enumerate(sentence):
		vec = _W2VEC.get(word, None)
		if vec is None:
			global miss
			miss += 1
			vec = np.random.rand(W2VEC_LEN)
		sent_vec[idx, :] = vec

	return sent_vec

def question_module(q):
	'Returns vec of a single question'
	q_vec = vectorize(q)
	q_vec = np.sum(q_vec, axis=0)

	return q_vec

def support_sent_module(sents):
	'Returns vec of the supporting sentences of a single story'
	sent_vec_sum = np.zeros(W2VEC_LEN)
	for sent in sents:
		sent_vec = vectorize(sent)
		sent_vec = np.sum(sent_vec, axis=0)
		sent_vec_sum += sent_vec    #Just Vector sum

	return sent_vec_sum

def execute_input(train):
	'Create a merged vector representation of both Question and Support Sentences'
	#Setup Input repr data for Answer Module
	N = len(train)
	X = np.zeros((N, 2*W2VEC_LEN))
	y = np.zeros((N, 2))
	YS = 0; NO = 1  #Yes - 0th index and No - I index
	
	for idx, each in enumerate(train):
		support_sents, q, a = each
		q_vec = question_module(q)
		support_sent_vec = support_sent_module(support_sents)

		# print q_vec.shape, support_sent_vec.shape  
		ip_vec = np.concatenate([q_vec, support_sent_vec])
		X[idx, :] = ip_vec
		
		if a[0] == 'no':
			i, j = NO, YS
		else: #YES
			i, j = YS, NO
		_y = np.zeros(2)
		_y[i] = 1; _y[j] = 0
		y[idx,:] = _y

	return X, y


def train_NN(X, y):
	#Train the answer module
	N = X.shape[0]
	batch_sz = 32
	epochs = 3
    
	val_id = int(N*0.8)
	nn(X[:val_id, :], y[:val_id, :], X[val_id:, :], y[val_id:,:])
	    # print get_weights()

def get_weights():
	return ans.l1.weight.data.numpy()#, ans.l2.weight.data.numpy(), ans.scores.weight.data.numpy()

def predict(X):
	N = X.shape[0]
	y = model.predict(X)
	return y

def main():
	#Fetch and Proprocess Data
	train, test = fetch_data()

	X, y = execute_input(train)

	train_NN(X, y)


if __name__ == "__main__":
	# setup_data()
	# print vectorize(["this", "is", "an", "example"])
	# print len(_W2VEC), _W2VEC['the']
	main()
