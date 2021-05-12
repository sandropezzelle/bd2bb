import os
import os.path
import bcolz
import pickle
import numpy as np

np.random.seed(0)

def load_glove_embeddings(data_directory,size,language_encoder):
    # load glove embeddings given data directory and embedding size
    if language_encoder == 'glove6B':
        path = 'glove.6B/'
    else:
        print('Embeddings not available. Please download them and try again')
        exit(0)
    typedim = '6B.'+str(size)

    if os.path.isdir(path + typedim + '.dat'):
        pass
    else:
        vectors, words, word2idx = prepare_glove_embeddings(path,typedim,size)
        pickle.dump(words, open(f'{path}/' + str(typedim) + '_words.pkl', 'wb'))
        pickle.dump(word2idx, open(f'{path}/' + str(typedim) +'_idx.pkl', 'wb'))
        print('built pkl files')

    if os.path.isfile(path + typedim + '_words.pkl'):
        pass
    else:
        print('Embeddings not found. Please check and try again!')
        exit(0)

    if os.path.isfile(path + typedim + '_idx.pkl'):
        pass
    else:
        print('Embeddings not found. Please check and try again!')
        exit(0)

    vectors = bcolz.open(f'{path}/' + typedim + '.dat')[:]
    words = pickle.load(open(f'{path}/' + typedim + '_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{path}/' + typedim + '_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}

    print('Language embeddings loaded!')

    return glove


def prepare_glove_embeddings(path_to_embeddings,typedim,size):
    # read embeddings from raw txt file

    #print(path_to_embeddings)
    print(path_to_embeddings + 'glove.' + str(typedim) + 'd.txt')

    if os.path.isfile(path_to_embeddings + 'glove.' + str(typedim) + 'd.txt'):
        pass
    else:
        print('Please download embeddings and try again!')
        exit(0)

    f = open(path_to_embeddings + 'glove.' + str(typedim) + 'd.txt', 'rb')
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{path_to_embeddings}/' + str(typedim) + '.dat', mode='w')

    words = []
    idx = 0
    word2idx = {}

    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400000, int(size))), rootdir=f'{path_to_embeddings}/' +
                                                                             str(typedim) + '.dat', mode='w')
    vectors.flush()

    return vectors, words, word2idx