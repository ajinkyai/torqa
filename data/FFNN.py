from __future__ import division
from data_utils import load_task
import csv
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F


W2VEC_LEN = 50
GLOVE_path = "./raw/glove.6B/glove.6B.50d.txt"
_W2VEC = None

class AnswerModule(torch.nn.Module):
    def __init__(self, D=2*W2VEC_LEN, H1=200, H2=200, C=2):
        """
        @param D:  int - Features in each input example
        @param H:  int - Hidden layer neurons
        @param C: int -  No. of classes
        """
        super(AnswerModule, self).__init__()
        torch.manual_seed(1)
        D = 2 * W2VEC_LEN
        C = 2
        self.l1 = torch.nn.Linear(D, H1, bias=True)
        self.l2 = torch.nn.Linear(H1, H2, bias=True)  
        self.scores = torch.nn.Linear(H2, C, bias=True) 

    def forward(self, X):
        h1 = self.l1(X).clamp(min=0)
        h2 = self.l2(h1)

        scores = self.scores(h2)
        probs = F.softmax(scores)   #Numerically stable by using logexp trick

        return probs

ans = AnswerModule(H1=400, H2=200)


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
	epochs = 20
	optimizer = torch.optim.Adam(params=ans.parameters())
    
	for _ in xrange(epochs):
	    idx = range(N)
	    # sz = (sz+300 if sz+300<N else sz)   #Increasing batch size every epoch
	    np.random.shuffle(idx)
	    _X = X[idx,:]
	    _y = y[idx,:]
	    for st in range(0, N, batch_sz):
	        end = st + batch_sz
	        batch_X = Variable(torch.from_numpy(_X[st:end])).float()
	        batch_y = Variable(torch.from_numpy(_y[st:end])).float()
	        
	        probs = ans(batch_X)
	        log_loss = -1 * torch.sum(torch.log(probs) * batch_y)/N
	        
	        log_loss.backward()  #Computes gradients
	        optimizer.step() #Updates weight params
	    print "Epoch:", _, ": ", log_loss.data[0]

def predict(X):
	N = X.shape[0]

	_X = Variable(torch.from_numpy(X), requires_grad=False).float()
	probs = ans(_X)
	probs = probs.data.numpy()
	# print probs.shape, probs[0:3,:]
	# return
	y_no = np.argmax(probs, axis=1)

	return y_no

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
