"""
Project: bd2bb
Sandro Pezzelle
University of Amsterdam
October 2020

inspired by Jabri et al. (2016). Revisiting Visual Question Answering Baselines

runs with "python evaluate.py"
"""

import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch import optim

from torch.autograd import Variable

from utils.load_dataset import load_data
from utils.extract_embeddings import load_glove_embeddings
from utils.visual_encoder import load_resnet152_features
from utils.visual_encoder import load_resnet101_features
from utils.prepare_inputs import bd2bb_data_loader

np.random.seed(0)

# print torch version
print('Using torch version:', torch.__version__)

# check if cuda is available
use_cuda = torch.cuda.is_available()
print('Cuda available:', use_cuda)


def evaluate(split_data,split_labels):
    myinp = split_data
    mylab = split_labels  # [i]
    mylab = mylab.reshape(len(split_labels), 1)
    y_pred = model(myinp)
    maxvalue, index = torch.max(y_pred, 1)
    pred = index.reshape(1, len(split_labels))[0]
    num_correct = torch.sum(pred == split_labels)  # a Tensor
    print(num_correct)
    acc = (num_correct.item() * 100.0 / len(split_labels))  # scalar
    loss = loss_fn(y_pred, mylab)
    return loss, acc


if __name__ == '__main__':
    """
    define arguments
    """
    parser = argparse.ArgumentParser()

    # read arguments
    parser.add_argument("-data_dir", type=str, default="../data/", help='Data Directory')
    parser.add_argument("-out_dir", type=str, default="./output/multimodal/21_epoch-34.pt", help='Output Directory')
    parser.add_argument("-manual_seed", type=int, default=42, help='Set your seed manually')
    parser.add_argument("-setting", type=str, default="multimodal", help='Model setting')
                        # only-language | only-vision | only-action
    parser.add_argument("-training_type", type=str, default="all-softmax", help='Training data sampling')
    parser.add_argument("-language_encoder", type=str, default="glove6B", help='Type of language encoder for I and A')
    parser.add_argument("-language_dim", type=int, default=300, help='Size of language embeddings')
    parser.add_argument("-n_epochs", type=int, default=50, help='Number of training epochs')
    parser.add_argument("-batch_size", type=int, default=32, help='Number of batches')
    parser.add_argument("-image_encoder", type=str, default="resnet101", help='Type of language encoder for V')
    parser.add_argument("-early_stopping", type=str, default="no", help='Yes: stops after 10 epochs w/ no improvements')
    parser.add_argument("-image_dim", type=int, default=2048, help='Size of visual embeddings')
    parser.add_argument("-fusion", type=str, default="concatenation", help='Type of multimodal fusion')
    parser.add_argument("-n_choices", type=int, default=5, help='Number of answers per each datapoint')
    parser.add_argument("-hdim", type=int, default=8192, help='Set dimensionality of classification hidden layer')
    parser.add_argument("-dropout", type=float, default=0.0, help='Set dropout classification')

    args = parser.parse_args()

    # print setting: multimodal, onlyL, onlyV
    setting = args.setting
    print('setting:',setting)

    # print manual seed to reproduce same results
    seed = args.manual_seed
    print('manual seed:',seed)
    torch.manual_seed(seed)

    # print training type
    training_type = args.training_type
    print('training_type:',training_type)

    # print type of language encoder: glove, etc.
    language_encoder = args.language_encoder
    print('language_encoder:', language_encoder)

    # print type of language encoder: resnet101, etc.
    image_encoder = args.image_encoder
    print('image_encoder:', image_encoder)

    # load best model
    out_path = args.out_dir
    print('loaded model:', out_path)
    modelpath = os.path.abspath(out_path)



    # check size of language representations
    imp_lang_dim = ['50','100','200','300']
    if str(args.language_dim) in imp_lang_dim:
        pass
    else:
        print('Language dimension not found. Available ones: 50, 100, 200, 300')
        exit(0)


    # set up type of language encoder (and load features)
    if language_encoder.startswith('glove'):
        embeddings = load_glove_embeddings(args.data_dir,args.language_dim,language_encoder)

    # set up type of visual encoder (and load features)
    if str(image_encoder).endswith('152'):
        trainlist, vallist, testlist, imgidxs = load_resnet152_features('./',image_encoder)
    elif str(image_encoder).endswith('101'):
        imgidxs, trainlist, vallist, testlist = load_resnet101_features('./',image_encoder)

    # load json dataset and split IDs
    dataset, trainIDs, valIDs, testIDs, EtestIDs, HtestIDs = load_data(args.data_dir)

    # define type and name of experiment
    experiment = setting + '-' + args.fusion
    print('Type of architecture:', experiment)


    # prepare inputs to be fed into model
    train_data, train_labels = bd2bb_data_loader(experiment,args.language_dim,args.image_dim,dataset,trainIDs,
                                                 embeddings,args.n_choices,trainlist,vallist,testlist,imgidxs,
                                                 args.training_type,setting)

    print('train data correctly encoded!')
    print('train data size:', train_data.size())

    val_data, val_labels = bd2bb_data_loader(experiment,args.language_dim,args.image_dim,dataset,valIDs,
                                             embeddings,args.n_choices,trainlist,vallist,testlist,imgidxs,
                                             args.training_type,setting)

    print('val data correctly encoded!')
    print('val data size:', val_data.size())

    test_data, test_labels = bd2bb_data_loader(experiment,args.language_dim,args.image_dim,dataset,testIDs,
                                               embeddings,args.n_choices,trainlist,vallist,testlist,imgidxs,
                                               args.training_type,setting)

    print('test data correctly encoded!')
    print('test data size:', test_data.size())


    if setting == 'multimodal':
        init_dim = int(args.image_dim)+int(args.language_dim)+int(args.language_dim)
    elif setting == 'only-language':
        init_dim = int(args.language_dim)+int(args.language_dim)
    elif setting == 'only-vision':
        init_dim = int(args.image_dim)+int(args.language_dim)
    elif setting == 'only-action':
        init_dim = int(args.language_dim)




    # build model
    hdim = int(args.hdim)

    model = nn.Sequential(
        nn.Linear(init_dim, hdim),
        nn.ReLU(),
        nn.Dropout(float(args.dropout)),
        nn.Linear(hdim,1),
        nn.Softmax(dim=1), # was 0
    )

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cpu')

    #with torch.no_grad():
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model.eval()

    lossV, accV = evaluate(val_data, val_labels)
    lossT, accT = evaluate(test_data, test_labels)
    # test_loss_hist.append(float(lossT.item()))
    # test_acc_hist.append(float(accT))

    print('val loss:', lossV.item())
    print('val accuracy:', accV)
    print('test loss:', lossT.item())
    print('test accuracy:', accT)
