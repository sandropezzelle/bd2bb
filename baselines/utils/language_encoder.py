import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
from torch.autograd import Variable
from torch import nn
from torch import autograd

np.random.seed(0)

tokenizer = get_tokenizer("basic_english")

def get_sentence_embedding(sentence, emb, landim):
    """
    get sentence embeddings per each Intention or Action
    """

    # sanity check to remove consecutive whitespaces
    sent = sentence.split()
    newsentence = " ".join(sent)
    target_vocab = tokenizer(newsentence)

    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, landim))

    for i, word in enumerate(target_vocab):
        try:
            # average of embeddings
            weights_matrix[i] = emb[word]
        except KeyError:
            # do not consider missing word
            pass

    average_representation = weights_matrix.sum(0)
    avg_rep = torch.from_numpy(average_representation)
    avg_sentence_rep = Variable(torch.Tensor(avg_rep.float()), requires_grad=False)

    return avg_sentence_rep