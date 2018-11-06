'''
Code taken from https://github.com/pytorch/examples/tree/master/word_language_model
'''

import os
import argparse
import pickle
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import time
import hyperparameters as hyp
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from scipy import sparse
import itertools
from sklearn import metrics

def create_emb_layer(matrix, padding_idx, non_trainable=False):

	num_embeddings, embedding_dim = matrix.shape
	padding_idx = torch.tensor(padding_idx).to(device)
	#initialize with vocabulary size in the training set and the dimension of each embedding vector
	emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

	#Add the pretrained embeddings to the layer
	emb_layer.weight.data.copy_(torch.from_numpy(matrix))
	# emb_layer.weight.requires_grad = True
	if non_trainable:
		emb_layer.weight.requires_grad = False

	return emb_layer, num_embeddings

def get_actual_scores(labels):
	# print labels
	#list to hold the label scores
	labels_batch_score = list()  #initialize
	# print label_to_id
	for label_list in labels:
		labels_score = label_size * [0]  #initialize
		# print "Labels are:"
		# print label_list	

		label_list = list(np.asarray(label_list))

		for i, label in enumerate(label_list):
			if (label == 0):
				labels_score[i] = float(label)
			else:
				# print label_to_id[label]
				# print label_to_id[label]
				labels_score[label_to_id[label]] = 1.0

		labels_batch_score.append(labels_score)

	return labels_batch_score

# # #Create a neural network with an embedding layer as first layer and a GRU layer
class Model(torch.nn.Module):
	def __init__(self, obj, vocab_limit, embedding_dim, hidden_dim, num_layers):
		super(Model, self).__init__()
		# self._size = 10
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		# self.batch_size = batch_size
		self.embeddings, num_embeddings = create_emb_layer(obj.train_weights_matrix, 0, False)

		# self.lstm_sent = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // 2, num_layers=num_layers, bidirectional=True, batch_first=False)
		self.lstm_char = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)

		self.lstm_sent = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)

		self.linear_sent_char = nn.Linear(self.hidden_dim*4, 100)

		# self.linear_labels = nn.Linear(label_size*self.embedding_dim, 20)

		self.linear_dense = nn.Linear(800, self.hidden_dim)
		self.linear_labels_correct = nn.Linear(self.hidden_dim, label_size)

		self.linear_labels_incorrect = nn.Linear(self.hidden_dim, label_size)

	def forward(self, input_sent, input_char, input_label, batch_size):

		# Set initial states for the sentence encoder
		self.h0_sent = torch.nn.Parameter(torch.randn(self.num_layers*2, batch_size, self.hidden_dim).to(device), requires_grad=True) # 2 for bidirection 
		self.c0_sent = torch.nn.Parameter(torch.randn(self.num_layers*2, batch_size, self.hidden_dim).to(device), requires_grad=True)

		# Set initial states for the character context encoder
		self.h0_char = torch.nn.Parameter(torch.randn(self.num_layers*2, batch_size, self.hidden_dim).to(device), requires_grad=True) # 2 for bidirection 
		self.c0_char = torch.nn.Parameter(torch.randn(self.num_layers*2, batch_size, self.hidden_dim).to(device), requires_grad=True)

		self.x_sent = self.embeddings(input_sent)
		self.x_char = self.embeddings(input_char)
		self.x_labels = self.embeddings(input_label)

		h_sent, (h_t_sent, c_t_sent) = self.lstm_sent(self.x_sent,(self.h0_sent, self.c0_sent))
		h_t_sent_forward = h_t_sent[2]
		h_t_sent_backward = h_t_sent[3]
		h_s = torch.cat((h_t_sent_forward, h_t_sent_backward), dim=-1)

		h_char, (h_t_char, c_t_char) = self.lstm_char(self.x_char,(self.h0_char, self.c0_char))
		h_t_char_forward = h_t_char[2]
		h_t_char_backward = h_t_char[3]
		h_c = torch.cat((h_t_char_forward, h_t_char_backward), dim=-1)
		h_e = torch.cat((h_s, h_c), 1)

		linear_h_e = F.relu(self.linear_sent_char(h_e))

		#reshape the encoder vector
		linear_h_e = linear_h_e.view(linear_h_e.size()[0], 1, -1)

		linear_h_e_repeat = linear_h_e.expand(linear_h_e.size()[0], self.x_labels.size()[1], linear_h_e.size()[2])

		elem_prod = self.x_labels * linear_h_e_repeat

		linear_concat = elem_prod.view(elem_prod.size()[0], 1, -1)
		
		#reshape linear concat
		linear_concat = F.relu(linear_concat.view(linear_concat.size()[0], linear_concat.size()[2]))

		dense_out = F.relu(self.linear_dense(linear_concat))

		labels_correct = self.linear_labels_correct(dense_out)
		labels_incorrect = self.linear_labels_incorrect(dense_out)

		labels_stacked = torch.stack([labels_correct, labels_incorrect], dim = -1)

		labels_softmax = torch.nn.functional.softmax(labels_stacked, dim = -1)

		return labels_softmax, self.x_labels

class Corpus(object):

	def __init__(self, path):
		self.path = path

	def get_data(self):
		#load the training data
		# self.train_sentences = pickle.load(open(os.path.join(path, "training_sentences_id.p"), "rb"))[:500]	
		# self.train_contexts = pickle.load(open(os.path.join(path, "training_char_contexts_id.p"), "rb"))[:500]
		# self.train_labels = pickle.load(open(os.path.join(path, "training_labels_id.p"), "rb"))[:500]
		self.train_sentences = pickle.load(open(os.path.join(self.path, "training_sentences_id.p"), "rb"))[:]
		self.train_contexts = pickle.load(open(os.path.join(self.path, "training_char_contexts_id.p"), "rb"))[:]
		self.train_labels = pickle.load(open(os.path.join(self.path, "training_labels_id.p"), "rb"))[:]
		self.train_word2idx = pickle.load(open(os.path.join(self.path, "training_word2idx.p"), "rb"))
		self.train_idx2word = pickle.load(open(os.path.join(self.path, "training_idx2word.p"), "rb"))
		self.train_weights_matrix = pickle.load(open(os.path.join(self.path, "training_glove_matrix.p"), "rb"))

		#load the validation data
		self.val_sentences = pickle.load(open(os.path.join(self.path, "validation_sentences_id.p"), "rb"))[:]
		self.val_contexts = pickle.load(open(os.path.join(self.path, "validation_char_contexts_id.p"), "rb"))[:]
		self.val_labels = pickle.load(open(os.path.join(self.path, "validation_labels_id.p"), "rb"))[:]		# self.val_word2idx = pickle.load(open(os.path.join(path, "validation_word2idx.p"), "rb"))

		#load the test data
		self.test_sentences = pickle.load(open(os.path.join(self.path, "test_sentences_id.p"), "rb"))[:]
		self.test_contexts = pickle.load(open(os.path.join(self.path, "test_char_contexts_id.p"), "rb"))[:]
		self.test_labels = pickle.load(open(os.path.join(self.path, "test_labels_id.p"), "rb"))[:]		# self.val_word2idx = pickle.load(open(os.path.join(path, "validation_word2idx.p"), "rb"))

		# self.val_idx2word = pickle.load(open(os.path.join(path, "validation_idx2word.p"), "rb"))
		# self.val_weights_matrix = pickle.load(open(os.path.join(path, "validation_glove_matrix.p"), "rb"))


###########################################################################################
# remove the zero padding module
def zero_padding(X):
	#X is the data that is being passed
	#get the length of each sentence text in sentences / char context / labels
	X_lengths = [len(sentence) for sentence in X]

	#create an empty matrix with padding tokens
	pad_token = 0
	longest_sent = max(X_lengths)
	batch_size = len(X)
	padded_X = np.ones((batch_size, longest_sent)) * pad_token

	#copy over the actual sequences 
	for i, x_len in enumerate(X_lengths):
		sequence = X[i]
		padded_X[i, 0:x_len] = sequence[:x_len]
	
	return padded_X

####Remove the map_to_id module#############################################################
def map_to_id(labels):
	label_to_id = dict()	
	id_to_label = dict()

	for label_list in labels:
		# print label_list

		for label in label_list:
			
			#if already in the dictionary, then do nothing
			# if label in label_to_id:
			try:
				label_to_id[label] 
			except Exception, e:
				label_to_id[label] = len(label_to_id)
	# print label_to_id

	for label in label_to_id:
		id_to_label[label_to_id[label]] = label

	return label_to_id, id_to_label

def trainModel(obj):

	losses = list()

	for i in range(hyp.n_epochs):
		#initialize the loss for each epoch
		total_loss = torch.Tensor([0])
		permutation = torch.randperm(train_data_size)

		for j in range(0, train_data_size, train_batch_size):
			batch_indices = permutation[j:j+train_batch_size]
			batch_sentences, batch_contexts, batch_actual_labels = obj.train_sentences[batch_indices], obj.train_contexts[batch_indices], obj.train_labels[batch_indices]
			# print "Batch size:"
			# print batch_indices.size()
			batch_labels = torch.tensor([labels_unique for k in range(batch_indices.size(0))])
			# print batch_labels
			# print
			batch_sentences = Variable(batch_sentences.long().to(device))
			batch_contexts = Variable(batch_contexts.long().to(device))
			batch_labels = Variable(batch_labels.long().to(device))

			# print "labels:"
			# print batch_actual_labels
			# print
			optimizer.zero_grad()  #clear all the gradients
			a = list(model.parameters())[0].clone()
			#forward pass
			# predicted_scores, predicted_labels, predicted_val_label_embeddings = model(batch_sentences, batch_contexts, batch_labels, len(batch_indices))
			##########################################################################################
			# print "Actual labels"

			# print batch_labels
			#########################################################################################
			predicted_scores, label_embeds = model(batch_sentences, batch_contexts, batch_labels, len(batch_indices))

			actual_scores = Variable(torch.tensor(get_actual_scores(batch_actual_labels)).to(device))

			wts = ((1 - weights["train"].to(device)) * actual_scores.to(device).float() + weights["train"].to(device) * (1 - actual_scores.to(device).float()))
			
			######Give same weight to each class##################################################################
			wts = torch.ones(batch_sentences.size()[0], 8).to(device)
			loss = loss_function(predicted_scores[:, :, 0], actual_scores)
			# print "Weihts:"
			# print wts

			loss.backward()

			optimizer.step()

			b = list(model.parameters())[0].clone()

			# print torch.equal(a.data, b.data)
	                total_loss += loss.item()

		        predicted_score_detach = Variable(predicted_scores.data, requires_grad=False)

		        labels_max_score = torch.max(predicted_score_detach, dim=-1)
	        	# print labels_max_score

	        	softmax_scores_ids = labels_max_score[1]
	        	# print softmax_scores_ids
            		batch_predict_label = []

			for k in range(softmax_scores_ids.size()[0]):
				line_predict_label = label_size * [0]
				#get each point in the minibatch
				zero_indices = (softmax_scores_ids[k] == 0).nonzero()

				for idx in zero_indices:
					line_predict_label[idx.cpu().numpy()[0]] = id_to_label[idx.cpu().numpy()[0]]
				batch_predict_label.append(line_predict_label)
				
		print('For epoch %d loss :%g'%(i, loss.data[0]))



#y_actual = actual labels
#y_hat = predicted labels
def perf_measure(y_actual, y_hat):
	TP = 0
	FP = 0
	TN = 0
	FN = 0

	for i in range(len(y_hat)): 
		if y_actual[i]==y_hat[i]==1:
			TP += 1
		if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
			FP += 1
		if y_actual[i]==y_hat[i]==0:
			TN += 1
		if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
			FN += 1
    
	print (TP, FP, TN, FN)
	
	return(TP, FP, TN, FN)

if torch.cuda.is_available():
    if not hyp.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if hyp.cuda else "cpu")

#######Load data##############################################

corpus = Corpus("../data/")

corpus.get_data()

#get only train and test data
corpus.train_sentences = corpus.train_sentences[:] + corpus.val_sentences[:]
corpus.train_contexts = corpus.train_contexts[:] + corpus.val_contexts[:]
corpus.train_labels = corpus.train_labels[:] + corpus.val_labels[:]


corpus.val_sentences = corpus.test_sentences[:]
corpus.val_contexts = corpus.test_contexts[:]
corpus.val_labels = corpus.test_labels[:]

train_data_size = len(corpus.train_sentences)
val_data_size = len(corpus.test_sentences)

#flatten the nested lists of labels
labels_unique = list(set(list(itertools.chain(*corpus.train_labels))))
print labels_unique
print [corpus.train_idx2word[label] for label in labels_unique]
# print labels_unique
#get the length of the label set
label_size = len(labels_unique)
# print label_size
label_to_id, id_to_label = map_to_id(corpus.train_labels)
print label_to_id

# for label_id in labels_unique:
	# print corpus.train_idx2word[label_id]
#specify the minibatch size for training
train_batch_size = hyp.train_batch_size

#specify the number of layers in the network
num_layers = hyp.num_layers




##############################zero pad the training and vaidation data##############################################################
corpus.train_sentences = zero_padding(corpus.train_sentences)
corpus.train_contexts = zero_padding(corpus.train_contexts)
corpus.train_labels = zero_padding(corpus.train_labels)
corpus.val_sentences = zero_padding(corpus.val_sentences)
corpus.val_contexts = zero_padding(corpus.val_contexts)
corpus.val_labels = zero_padding(corpus.val_labels)
#####################################################################################################################

########################convert the dataset to pytorch tensors #######################################################
corpus.train_sentences = torch.tensor(corpus.train_sentences, dtype=torch.int32)
corpus.train_contexts = torch.tensor(corpus.train_contexts, dtype=torch.int32)
corpus.train_labels = torch.tensor(corpus.train_labels, dtype=torch.int32)
corpus.val_sentences = torch.tensor(corpus.val_sentences, dtype=torch.int32)
corpus.val_contexts = torch.tensor(corpus.val_contexts, dtype=torch.int32)
corpus.val_labels = torch.tensor(corpus.val_labels, dtype=torch.int32)
#######################################################################################################################

model = Model(corpus, hyp.vocabLimit, hyp.embedding_dim, hyp.hidden_units, hyp.num_layers)


# print "#################################################################################################################################"
if torch.cuda.is_available():
	model.cuda()

loss_function = nn.BCELoss()

optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=hyp.lr)

# for name, p in model.named_parameters():
#     if "weight" in name:
#         print name
#         nn.init.xavier_normal(p)

weights = pickle.load(open("scs-baselines/class_weights.p", "rb"))
#training the model
trainModel(corpus)
# print "Embeddings are:"
# print changed_embeddings.size()
# print
#get the embeddings matrix after the training
embeddings_matrix = model.embeddings

batch_labels = [label_to_id.keys()]
batch_labels = Variable(torch.tensor(batch_labels))
# print "embeddings matrix:"
# print embeddings_matrix
# print "Labels:"
# print label_to_id.keys()
# print batch_labels
# print embeddings_matrix[batch_labels]
# print "The model parameters with no gradient"
for name, p in model.named_parameters():
    if p.grad is None:
        print name
        print p.grad
# for name, p in model.named_parameters():
#     if p.requires_grad:
#         print name

#Testing the model
# Test the model
with torch.no_grad():
	# eval_batch_size = corpus.val_labels.size(0)
	eval_batch_size = hyp.eval_batch_size
	print "Eval batch size:"
	print eval_batch_size
	print "Val data size:"
	print val_data_size
	correct = 0
	total = 0

	# for i in range(corpus.val_sentences.size()[0]):
		# print i

	corpus.val_sentences = Variable(corpus.val_sentences.long().to(device))
	corpus.val_contexts = Variable(corpus.val_contexts.long().to(device))
	corpus.val_labels = Variable(corpus.val_labels.long().to(device))

	permutation = torch.randperm(val_data_size)

	for k in range(0, val_data_size, eval_batch_size):

		sum_TP = 0
		sum_FP = 0
		sum_TN = 0
		sum_FN = 0

		batch_indices = permutation[k:k+eval_batch_size]
		# print "The batch size is:"
		# print batch_indices.size(0)

		batch_val_sentences, batch_val_contexts, batch_val_actual_labels = corpus.val_sentences[batch_indices], corpus.val_contexts[batch_indices], corpus.val_labels[batch_indices]
		# print "Actual labels:"
		# print batch_val_actual_labels
		# print
		# print "Batch size:"
		# print batch_indices.size()
		#taking all the labels occurring in the dataset
		batch_val_labels = torch.tensor([labels_unique for l in range(batch_indices.size(0))])
		# print batch_val_labels.size()
		batch_val_labels = Variable(batch_val_labels.long().to(device))

		predicted_val_scores, label_val_embeds = model(batch_val_sentences, batch_val_contexts, batch_val_labels, batch_val_labels.size(0))

		# print "Predicted Val scores are:"
		# print predicted_val_scores
		# print
		# #Detach from the computation graph
		predicted_val_score_detach = Variable(predicted_val_scores.data, requires_grad=False)

		# #Get the max of the labels being correct and labels not being correct
		labels_val_max_score = torch.max(predicted_val_score_detach, dim=-1)

		# print labels_val_max_score
		# print
		# #Get the label ids which are max - whether it's label being present or label not being present
		softmax_val_scores_ids = labels_val_max_score[1]

		#get the prediction label
		batch_val_predict_label = []

		for i in range(softmax_val_scores_ids.size()[0]):
			line_predict_label = label_size * [0]
			# print softmax_val_scores_ids[i]
			#get each point in the minibatch
			zero_indices = (softmax_val_scores_ids[i] == 0).nonzero()
			# print zero_indices
			
			for idx in zero_indices:
				line_predict_label[idx.cpu().numpy()[0]] = id_to_label[idx.cpu().numpy()[0]]
				# print line_predict_label
				# print
			batch_val_predict_label.append(line_predict_label)
			
		batch_val_predict_label = np.array(batch_val_predict_label)

		
		predicted_val_labels = Variable(torch.tensor(batch_val_predict_label[:]), requires_grad=False)
		
		print "The predicted labels"
		print predicted_val_labels
		print
		print "----------------------------------------------------------------------------------------------------------"
		# #dictionary which holds the number of predicted and actual labels 
		label_predicted_actual = dict()

		# initialize with zero filled lists for each label
		# print corpus.train_idx2word
		for label in label_to_id:
			#first list has predicted labels, second has actual labels
			# print [predicted_val_labels.size()[0] * [0], predicted_val_labels.size()[0] * [0]]
			# print predicted_val_labels.size()[0]
			# print predicted_val_labels.size()[0]
			#for each label, predict for each label
			label_predicted_actual[label] = [predicted_val_labels.size()[0] * [0], predicted_val_labels.size()[0] * [0]]
			
		#add predicted and actual labels
		for i in range(batch_val_actual_labels.size()[0]):
			# print "Predicted:"
			# print predicted_val_labels[i]
			# print "Actual:"
			# print corpus.val_labels[i]
			# print
			# print "Value of j is:", j
			for j in range(predicted_val_labels[i].size()[0]):
				#ignore the zero value ==> no label predicted here
				if int(predicted_val_labels[i][j].cpu().numpy()) != 0:
					label_predicted_actual[int(predicted_val_labels[i][j].cpu().numpy())][0][i] = 1

			for j in range(batch_val_actual_labels[i].size()[0]):
				
				#ignore the zero padding
				if int(batch_val_actual_labels[i][j].cpu().numpy()) != 0:
					label_predicted_actual[corpus.train_word2idx[corpus.train_idx2word[int(batch_val_actual_labels[i][j].cpu().numpy())]]][1][i] = 1

		# print label_predicted_actual
		label_predicted_actual_performance = {'precision' : 0, 'recall' : 0, 'f1' : 0}



		for label in label_predicted_actual:
			print "For label:", corpus.train_idx2word[label]
			precision, recall, fbeta_score, support = precision_recall_fscore_support(sparse.csr_matrix(np.asarray(label_predicted_actual[label][1])), sparse.csr_matrix(np.asarray(label_predicted_actual[label][0])), average='samples')

			print "precision:", precision
			print "recall:", recall
			print "fbeta_score:", fbeta_score
			print "support:", support

			TP, FP, TN, FN = perf_measure(label_predicted_actual[label][1], label_predicted_actual[label][0])
			sum_TP += TP
			sum_FP += FP
			sum_TN += TN
			sum_FN += FN
			print sum_TP
			print sum_FP
			print sum_TN
			print sum_FN
			print

		micro_precision = float(sum_TP) / (sum_TP + sum_FP)
		micro_recall = float(sum_TP) / (sum_TP + sum_FN)
		micro_f1 = 2.0 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
		"The micro-average measures are:"
		print micro_precision
		print micro_recall
		print micro_f1
		print
		
		print type(label_val_embeds)
		print label_val_embeds.size()

		pickle.dump(label_val_embeds.data.cpu().numpy(), open("label_embeddings_trained.p", "wb"))

		# #get the micr-average precision, recall and f1 measure

		# average_precision = metrics.average_precision_score(batch_val_actual_labels.data.cpu().numpy(), predicted_val_labels.data.cpu().numpy(),
  #                                                    average="micro")

		# average_recall = metrics.average_recall_score(batch_val_actual_labels.data.cpu().numpy(), predicted_val_labels.data.cpu().numpy(),
  #                                                    average="micro")

		# average_f1 = metrics.f1(batch_val_actual_labels.data.cpu().numpy(), predicted_val_labels.data.cpu().numpy(),
  #                                                    average="micro")

		# print "Average Precision:"
		# print average_precision
		# print "Average Recall:"
		# print average_recall
		# print "Average f1:"
		# print average_f1





