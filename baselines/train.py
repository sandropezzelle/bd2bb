"""
Project: bd2bb
Sandro Pezzelle
University of Amsterdam
October 2020

inspired by Jabri et al. (2016). Revisiting Visual Question Answering Baselines

runs with "python train.py"
"""

import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch import optim

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
    mylab = split_labels
    mylab = mylab.reshape(len(split_labels), 1)
    y_pred = model(myinp)
    maxvalue, index = torch.max(y_pred, 1)
    pred = index.reshape(1, len(split_labels))[0]
    num_correct = torch.sum(pred == split_labels)  # a Tensor
    acc = (num_correct.item() * 100.0 / len(split_labels))  # scalar
    loss = loss_fn(y_pred, mylab)
    return loss, acc


def shuffle_train(traindata, trainlabels, num_classes):
    """
    function to shuffle train data
    """
    labels = trainlabels.float()
    labels = labels.reshape(len(labels), 1)
    one_hot_target = (labels == torch.arange(num_classes).reshape(1, num_classes)).float()
    one_hot = one_hot_target.reshape(len(labels),num_classes, 1)

    conc_input = torch.cat((traindata, one_hot), 2)
    shuf = conc_input[torch.randperm(conc_input.size()[0])]
    return shuf

def donotshuffle_valtes(valtestdata, valtestlabels, num_classes):
    labels = valtestlabels.float()
    labels = labels.reshape(len(labels), 1)
    one_hot_target = (labels == torch.arange(num_classes).reshape(1, num_classes)).float()
    one_hot = one_hot_target.reshape(len(labels),num_classes, 1)

    conc_input = torch.cat((valtestdata, one_hot), 2)
    return conc_input



def get_batch(splitlabels, num_classes, batch_size, start, shuf):
    """
    function to manually create batches
    """
    end = start + batch_size
    if end > len(splitlabels):
        inp = shuf[start:, :, :init_dim]
        lab = shuf[start:, :, init_dim:]
    else:
        inp = shuf[start:end, :, :init_dim]
        lab = shuf[start:end, :, init_dim:]

    labR = lab.reshape(len(lab), num_classes)
    p = ((labR == 1).nonzero())
    finalL = p[:,1:].reshape(len(lab))
    return inp, finalL

def get_number_parameters(model):
    """
    function to get number of model parameters
    """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp



if __name__ == '__main__':
    """
    define arguments
    """
    parser = argparse.ArgumentParser()
    # read arguments
    parser.add_argument("-data_dir", type=str, default="../data/", help='Data Directory')
    parser.add_argument("-out_dir", type=str, default="./output/", help='Data Directory')
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

    # print manual seed to reproduce results
    seed = args.manual_seed
    print('manual seed:',seed)
    torch.manual_seed(seed)

    # print type of language encoder
    language_encoder = args.language_encoder
    print(language_encoder)

    # print type of image encoder
    image_encoder = args.image_encoder
    print(image_encoder)

    # check if GloVe embeddings are present in the directory:
    if not os.path.exists('glove.6B'):
        print('Please download and unzip GloVe embeddings first! See Readme file')
        exit(0)
    else:
        print('GloVe embeddings found!')

    # make output directory
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if not os.path.exists(out_dir+str(setting)):
        os.mkdir(out_dir+str(setting))

    mypath = out_dir+str(setting)
    path2output = os.path.abspath(mypath)

    # create log file txt
    path2out = 'output/' + str(setting) + '/' + str(setting) + '_' + str(args.training_type) + '_seed' + \
               str(args.manual_seed) + '_hdim' + str(args.hdim) + '.txt'
    output = open(path2out, 'w')
    output.write('epoch,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc,Etest_loss,Etest_acc,Htest_loss,'
                 'Htest_acc'+'\n')


    # check size of pretrained language representations
    imp_lang_dim = ['50','100','200','300']
    if str(args.language_dim) in imp_lang_dim:
        pass
    else:
        print('Language dimension not found. Available ones: 50, 100, 200, 300')
        exit(0)

    # set up type of language encoder and load features
    if language_encoder.startswith('glove'):
        # embeddings = load_glove_embeddings(data_dir,args.language_dim,language_encoder)
        embeddings = load_glove_embeddings('./',args.language_dim,language_encoder)
    else:
        print('Not implemented yet')
        exit(0)

    # define type of experiment, e.g. multimodal-concatenation
    experiment = setting + '-' + args.fusion
    print('Type of architecture:', experiment)

    # set up type of visual encoder (and load features)
    if str(image_encoder).endswith('152'):
        trainlist, vallist, testlist, imgidxs = load_resnet152_features('./',image_encoder)
    elif str(image_encoder).endswith('101'):
        imgidxs, trainlist, vallist, testlist = load_resnet101_features('./',image_encoder)


    # load json dataset and split IDs
    dataset, trainIDs, valIDs, testIDs, EtestIDs, HtestIDs = load_data(args.data_dir)

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


    # work with batches
    batch_size = args.batch_size

    n_batchesT = round(len(train_labels) / batch_size)
    n_batchesV = round(len(val_labels) / batch_size)
    n_batchesTe = round(len(val_labels) / batch_size)

    shufT = shuffle_train(train_data, train_labels, args.n_choices)
    notshufV = donotshuffle_valtes(val_data,val_labels,args.n_choices)
    notshufTe = donotshuffle_valtes(test_data,test_labels,args.n_choices)


    # build model
    hdim = int(args.hdim)
    model = nn.Sequential(
        nn.Linear(init_dim, hdim),
        nn.ReLU(),
        nn.Dropout(float(args.dropout)),
        nn.Linear(hdim,1),
        nn.Softmax(dim=1),
    )
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # get_number_parameters(model)
    print("Number of parameters: {}".format(get_number_parameters(model)))

    epochs = 0
    previous = [0]
    train_acc_hist, train_loss_hist = [], []
    val_acc_hist, val_loss_hist = [], []
    test_acc_hist, test_loss_hist = [], []
    Etest_acc_hist, Etest_loss_hist = [], []
    Htest_acc_hist, Htest_loss_hist = [], []

    for t in range(args.n_epochs):
        start = 0
        optimizer.zero_grad()
        num_correct = 0
        loss = 0
        for b in range(n_batchesT):
            inp, finalL = get_batch(train_labels, args.n_choices, batch_size, start, shufT)

            myinp = inp
            mylab = finalL
            mylab = mylab.reshape(len(finalL),1)

            y_pred = model(myinp)

            maxvalue, index = torch.max(y_pred, 1)
            pred = index.reshape(1, len(finalL))[0]

            num_correct += torch.sum(pred == finalL)  # a Tensor

            loss += loss_fn(y_pred, mylab)

            start += batch_size

        loss.backward()
        optimizer.step()

        Floss = loss.item() / n_batchesT
        acc = (num_correct.item() * 100.0 / len(train_labels))  # scalar
        print(t, 'training_loss:', Floss)
        print('training accuracy:', acc)
        train_loss_hist.append(float(Floss))  # .item()))
        train_acc_hist.append(float(acc))

        with torch.no_grad():
            startV = 0
            num_correctV = 0
            lossV = 0
            for b in range(n_batchesV):
                inpV, finalV = get_batch(val_labels, args.n_choices, batch_size, startV, notshufV)

                myinpV = inpV
                mylabV = finalV
                mylabV = mylabV.reshape(len(finalV), 1)

                y_predV = model(myinpV)

                maxvalueV, indexV = torch.max(y_predV, 1)
                predV = indexV.reshape(1, len(finalV))[0]

                num_correctV += torch.sum(predV == finalV)  # a Tensor

                lossV += loss_fn(y_predV, mylabV)

                startV += batch_size

            lossV = lossV.item() / n_batchesV
            accV = (num_correctV.item() * 100.0 / len(val_labels))  # scalar
            print('validation_loss:', lossV)
            print('validation accuracy:', accV)

        with torch.no_grad():
            startTe = 0
            num_correctTe = 0
            lossTe = 0
            for b in range(n_batchesTe):
                inpTe, finalTe = get_batch(test_labels, args.n_choices, batch_size, startTe, notshufTe)

                myinpTe = inpTe
                mylabTe = finalTe
                mylabTe = mylabTe.reshape(len(finalTe), 1)

                y_predTe = model(myinpTe)

                maxvalueTe, indexTe = torch.max(y_predTe, 1)
                predTe = indexTe.reshape(1, len(finalTe))[0]

                num_correctTe += torch.sum(predTe == finalTe)  # a Tensor

                lossTe += loss_fn(y_predTe, mylabTe)

                startTe += batch_size

            lossTe = lossTe.item() / n_batchesTe
            accTe = (num_correctTe.item() * 100.0 / len(test_labels))  # scalar
            test_loss_hist.append(float(lossTe))
            test_acc_hist.append(float(accTe))
            print('test_loss:', lossTe)
            print('test accuracy:', accTe)


            output.write(str(t) + ',' + str(Floss) + ',' + str(acc) + ',' + str(lossV) + \
                         ',' + str(accV) + ',' + str(lossTe) + ',' + str(accTe) + ',' + '\n')

            if args.early_stopping == 'yes':
                if float(accV) < float(max(val_acc_hist)):
                    epochs += 1
                else:
                    epochs = 0
                if epochs == 10:
                    print('Val accuracy did not increase for 10 epochs')
                    break

            pt = str(setting) + '_' + str(args.training_type) + '_seed' + str(args.manual_seed) + '_hdim' + str(args.hdim)

            if t == 0:
                print('Saving model first epoch')
                torch.save(model.state_dict(), os.path.join(path2output, 'epoch-{}.pt'.format(t)))

            # print(float(accV), float(max(val_acc_hist)))
            elif float(accV) > float(max(val_acc_hist)):
                print('Saving best model at this epoch')
                torch.save(model.state_dict(), os.path.join(path2output, 'epoch-{}.pt'.format(t)))

            val_loss_hist.append(float(lossV))
            val_acc_hist.append(float(accV))

    bestacc = int(np.argmax(val_acc_hist))

    print('epoch best acc:', bestacc)
    print('train acc at best epoch:', train_acc_hist[bestacc])
    print('val acc at best epoch:', val_acc_hist[bestacc])
    # print('val loss at best epoch:', val_loss_hist[bestacc])
    print('test acc at best epoch:', test_acc_hist[bestacc])

    if Etest_acc_hist != []:
        print('easy test acc at best epoch:', Etest_acc_hist[bestacc])
    if Htest_acc_hist != []:
        print('hard test acc at best epoch:', Htest_acc_hist[bestacc])

    output.close()
