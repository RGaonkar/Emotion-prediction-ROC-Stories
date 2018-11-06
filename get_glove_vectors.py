# import bcolz
import pickle
import numpy as np

words = []
idx = 0
word2idx = {}

#####Bcolz implementation#################################################################################
##########################################################################################################
# vectors = bcolz.carray(np.zeros(1), rootdir='../data/glove.6B.100d.dat', mode='w')

# with open('../data/glove.6B.100d.txt', 'rb') as f:

# 	for l in f:
# 		line = l.decode("utf-8").split()
# 		word = line[0]
# 		words.append(word)
# 		word2idx[word] = idx
# 		idx += 1
# 		vect = np.array(line[1:]).astype(np.float)
# 		vectors.append(vect)

# vectors = bcolz.carray(vectors[1:].reshape((400000, 100)), rootdir='../data/glove.6B.100d.dat', mode='w')
# vectors.flush()
##########################################################################################################
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

glove_dict = loadGloveModel("../../data_for_code/glove.6B.100d.txt")

pickle.dump(glove_dict, open('../../data_for_code/glove.6B.100d.dat', 'wb'))
pickle.dump(words, open('../../data_for_code/glove.6B.100_words.pkl', 'wb'))
pickle.dump(word2idx, open('../../data_for_code/glove.6B.100_idx.pkl', 'wb'))

#using the above objects, create a dictionary that given a word returns its vector
# vectors = bcolz.open('../data/glove.6B.100d.dat')[:]
# words = pickle.load(open('../data/glove.6B.100_words.pkl', 'rb'))
# word2idx = pickle.load(open('../data/glove.6B.100_idx.pkl', 'rb'))

#dictionary comprehension FTW!
# glove = {w: vectors[word2idx[w]] for w in words}

print (glove_dict['the'])


