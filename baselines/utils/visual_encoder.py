import numpy as np
import json
import h5py

np.random.seed(0)

def load_resnet152_features(data_dir, image_encoder):

    # load pre-computed features
    resnet_path = str(data_dir) + str(image_encoder) + '/'

    feats = resnet_path+"ResNet_avg_image_features.h5"
    indexes = resnet_path+"ResNet_avg_image_features2id.json"

    idxs = open(indexes,'r')
    dataidxs = json.load(idxs)

    f = h5py.File(feats, "r")
    print("Keys: %s" % f.keys())

    testdata = list(f.keys())[0]
    traindata = list(f.keys())[1]
    valdata = list(f.keys())[2]

    # these are lists of vectors
    testlist = list(f[testdata])
    trainlist = list(f[traindata])
    vallist = list(f[valdata])

    return trainlist, vallist, testlist, dataidxs


def load_resnet101_features(data_dir, image_encoder):

    # load pre-computed features
    resnet_path = str(data_dir) + str(image_encoder) + '/'
    # resnet_path = str(data_dir.split('/')[0])+'/'+str(data_dir.split('/')[1])+str(image_encoder)+'/'
    indexes = resnet_path+"features_resnet101.json"

    idxs = open(indexes,'r')
    dataidxs = json.load(idxs)

    trainlist, vallist, testlist = [],[],[]

    return dataidxs, trainlist, vallist, testlist


def get_visual_features(url,featlistTr,featlistV,featlistTe,json_indexes,imgdim):
    # for each image (url), get visual features
    if featlistV == [] and featlistTr == [] and featlistTe == []:
        """
        get ResNet101
        """
        code = str(url).split('.')[0]
        vv = json_indexes.get(code, "")
        vvv = np.asarray(vv)

    else:
        """
        get ResNet152
        """
        if url in json_indexes['train2id']:
            vv = json_indexes['train2id'][url]
            splitF = 'trainF'
        elif url in json_indexes['val2id']:
            vv = json_indexes['val2id'][url]
            splitF = 'valF'
        elif url in json_indexes['test2id']:
            vv = json_indexes['test2id'][url]
            splitF = 'testF'
        else:
            splitF = 'none'

        if splitF == 'trainF':
            vvv = featlistTr[vv]
        elif splitF == 'valF':
            vvv = featlistV[vv]
        elif splitF == 'testF':
            vvv = featlistTe[vv]
        elif splitF == 'none':
            vvv = np.random.rand(1,imgdim)

    vis_feats = vvv

    return vis_feats