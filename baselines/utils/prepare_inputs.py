import torch
from utils.language_encoder import get_sentence_embedding
from utils.visual_encoder import get_visual_features
from torch.autograd import Variable
import random
from random import randint

random.seed(0)

def bd2bb_data_loader(experiment_name,languagedim,imagedim,dataset,IDs,embeddings,
                      nchoices,TimgL,VimgL,TTimgL,imgidxs,training_type,setting):
    # prepare inputs based on type of experiment
    # and input size

    if experiment_name == 'only-language-concatenation':
        split = prepare_conc(languagedim,imagedim,dataset,IDs,embeddings,
                             nchoices,TimgL,VimgL,TTimgL,imgidxs,training_type,setting)

    elif experiment_name == 'only-language-sum':
        print('To be implemented')
        exit(0)
    elif experiment_name == 'only-language-product':
        print('Product fusion only available in setting multimodal')
        exit(0)

    elif experiment_name == 'only-vision-concatenation':
        split = prepare_conc(languagedim,imagedim,dataset,IDs,embeddings,
                             nchoices,TimgL,VimgL,TTimgL,imgidxs,training_type,setting)

    elif experiment_name == 'only-vision-sum':
        print('To be implemented')
        exit(0)
    elif experiment_name == 'only-language-product':
        print('Product fusion only available in setting multimodal')
        exit(0)

    elif experiment_name == 'multimodal-concatenation':
        split = prepare_conc(languagedim,imagedim,dataset,IDs,embeddings,
                             nchoices,TimgL,VimgL,TTimgL,imgidxs,training_type,setting)

    elif experiment_name == 'multimodal-sum':
        print('To be implemented')
        exit(0)
    elif experiment_name == 'multimodal-product':
        print('To be implemented')
        exit(0)

    elif experiment_name == 'only-action-concatenation':
        split = prepare_conc(languagedim,imagedim,dataset,IDs,embeddings,
                             nchoices,TimgL,VimgL,TTimgL,imgidxs,training_type,setting)

    return split


def prepare_conc(languagedim,imagedim,dataset,IDs,embeddings,nchoices,TimgL,VimgL,TTimgL,imgidxs,training_type,setting):

    imgM, qM, all_ansM = prepare_inputs(languagedim,imagedim,dataset,IDs,embeddings,
                                          nchoices,TimgL,VimgL,TTimgL,imgidxs)


    idxs = {}
    path2idx = open('../data/BD2BB.csv','r')
    content = path2idx.readlines()
    for line in content:
        midx = line.strip().split(',')[0]
        mval = line.strip().split(',')[-1]
        if str(midx) != 'id':
            if midx not in idxs:
                idxs[midx] = mval
            # idxs[midx].append(mval)
    idxslist = []
    for el in IDs:
        myv = int(idxs[el])
        idxslist.append(myv)

    # double-check distribution of classes
    # print(len(idxslist))
    # myfreq = {i: idxslist.count(i) for i in set(idxslist)}
    # print(myfreq)

    labsfromfile = torch.FloatTensor(idxslist)


    if len(IDs) == 2102: # TODO make this less hardcoded
        print('This is the training set!')
        split = 'training'
    else:
        split = 'other'

    if split == 'training':

        total_size = len(IDs)

        print('Training type:', training_type)
        print('Total size:', total_size)

        img_feats_b = torch.Tensor(total_size, nchoices, imagedim)  # was Tensor
        q_feats_b = torch.Tensor(total_size, nchoices, languagedim)  # was Tensor
        ans_feats_b = torch.Tensor(total_size, nchoices, languagedim)
        labels_b = labsfromfile.long()

        for i in range(len(labels_b)):  # recall that these are random !
            mypos = [0, 1, 2, 3, 4]
            mypos.remove(int(labels_b[i].item()))
            random.shuffle(mypos)
            for j in range(nchoices):
                if j == 0:
                    img_feats_b[i][labels_b[i].item()] = imgM[i]
                    q_feats_b[i][labels_b[i].item()] = qM[i]
                    ans_feats_b[i][labels_b[i].item()] = all_ansM[i][j]
                else:
                    myid = mypos[j - 1]
                    img_feats_b[i][myid] = imgM[i]
                    q_feats_b[i][myid] = qM[i]
                    ans_feats_b[i][myid] = all_ansM[i][j]


        v = Variable(img_feats_b, requires_grad=False)
        q = Variable(q_feats_b, requires_grad=False)
        a = Variable(ans_feats_b, requires_grad=False)
        # in val/test, l encodes all 0 (position of correct answer)
        l = Variable(labels_b, requires_grad=False)

        final_data, totsize = slice_input(v,q,a,setting,training_type,split)
        # conc_input = torch.cat((v, q, a), 2)



    else:
        total_size = len(IDs)

        img_feats_b = torch.Tensor(total_size, nchoices, imagedim)  # was Tensor
        q_feats_b = torch.Tensor(total_size, nchoices, languagedim)  # was Tensor
        ans_feats_b = torch.Tensor(total_size, nchoices, languagedim)
        labels_b = labsfromfile.long()

        for i in range(total_size):
            mypos = [0, 1, 2, 3, 4]
            mypos.remove(int(labels_b[i].item()))
            random.shuffle(mypos)
            for j in range(nchoices):
                if j == 0:
                    img_feats_b[i][labels_b[i].item()] = imgM[i]
                    q_feats_b[i][labels_b[i].item()] = qM[i]
                    ans_feats_b[i][labels_b[i].item()] = all_ansM[i][j]
                else:
                    myid = mypos[j - 1]
                    img_feats_b[i][myid] = imgM[i]
                    q_feats_b[i][myid] = qM[i]
                    ans_feats_b[i][myid] = all_ansM[i][j]

        v = Variable(img_feats_b, requires_grad=False)
        q = Variable(q_feats_b, requires_grad=False)
        a = Variable(ans_feats_b, requires_grad=False)
        l = Variable(labels_b, requires_grad=False)

        final_data, totsize = slice_input(v, q, a, setting, training_type, split)
        # conc_input = torch.cat((v, q, a), 2)

    return final_data, l



def slice_input(v,q,a,setting,training_type,split):

    if training_type != 'all-softmax' and split == 'training':
        if setting == 'multimodal':
            conc_input = torch.cat((v, q, a), 1)
        elif setting == 'only-language':
            conc_input = torch.cat((q, a), 1)
        elif setting == 'only-vision':
            conc_input = torch.cat((v, a), 1)
        elif setting == 'only-action':
            conc_input = a
        totsize = conc_input.size()[1]

    else:
        if setting == 'multimodal':
            conc_input = torch.cat((v, q, a), 2)
        elif setting == 'only-language':
            conc_input = torch.cat((q, a), 2)
        elif setting == 'only-vision':
            conc_input = torch.cat((v, a), 2)
        elif setting == 'only-action':
            conc_input = a
        totsize = conc_input.size()[2]
    return conc_input, totsize



def prepare_sum(languagedim,imagedim,dataset,IDs,embeddings,nchoices,TimgL,VimgL,TTimgL,imgidxs,training_type):

    imgM, qM, all_ansM = prepare_inputs(languagedim,imagedim,dataset,IDs,embeddings,
                                          nchoices,TimgL,VimgL,TTimgL,imgidxs)

    return sum_input

def prepare_prod(languagedim,imagedim,dataset,IDs,embeddings,nchoices,TimgL,VimgL,TTimgL,imgidxs,training_type):

    imgM, intM, all_ansM = prepare_inputs(languagedim,imagedim,dataset,IDs,embeddings,
                                          nchoices,TimgL,VimgL,TTimgL,imgidxs)

    return prod_input


def prepare_inputs(languagedim, imagedim, dataset, IDs, embeddings, nchoices, TimgL, VimgL, TTimgL, imgidxs):

    img_feats = torch.Tensor(len(IDs),imagedim) # train 7185, 2048
    q_feats = torch.Tensor(len(IDs), languagedim) # train 7185, 300
    all_ans_feats = torch.Tensor(len(IDs), nchoices, languagedim) # 7185, 5, 300

    c = 0
    for el in IDs:

        intention = str(dataset[el][0]['target_item']['intention'])
        INT_emb = get_sentence_embedding(intention, embeddings, languagedim)
        q_feats[c] = INT_emb

        target_act = str(dataset[el][0]['target_item']['action'])
        TA_emb = get_sentence_embedding(target_act, embeddings, languagedim)
        all_ans_feats[c][0] = TA_emb

        L1_decoy = str(dataset[el][0]['lang_decoy1']['action'])
        L1_emb = get_sentence_embedding(L1_decoy, embeddings, languagedim)
        all_ans_feats[c][1] = L1_emb

        L2_decoy = str(dataset[el][0]['lang_decoy2']['action'])
        L2_emb = get_sentence_embedding(L2_decoy, embeddings, languagedim)
        all_ans_feats[c][2] = L2_emb

        V1_decoy = str(dataset[el][0]['vis_decoy1']['action'])
        V1_emb = get_sentence_embedding(V1_decoy, embeddings, languagedim)
        all_ans_feats[c][3] = V1_emb

        V2_decoy = str(dataset[el][0]['vis_decoy2']['action'])
        V2_emb = get_sentence_embedding(V2_decoy, embeddings, languagedim)
        all_ans_feats[c][4] = V2_emb

        image = str(dataset[el][0]['target_item']['image_url'])
        url = image.split('/')[-1]
        feats = get_visual_features(url, TimgL, VimgL, TTimgL, imgidxs, imagedim)
        img_feats[c] = Variable(torch.Tensor(torch.from_numpy(feats).float()), requires_grad=False)  # needed?)

        c += 1

    return img_feats, q_feats, all_ans_feats