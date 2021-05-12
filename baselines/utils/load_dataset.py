import glob
import os
import json

def load_data(data_directory):
    # load json dataset with image urls, intentions, and actions
    myjson = []
    print(data_directory)
    os.chdir(data_directory)
    for file in glob.glob("*.json"):
        myjson.append(file)

    # check if there is only one json data file
    if len(myjson) == 1:
        myjsonfile = myjson[0]
    else:
        print('More than one json file in the folder!')
        exit(0)
    json_data = open(myjsonfile, 'r')
    data = json.load(json_data)

    # load split IDs
    mytxt = []
    for file in glob.glob("*.csv"):
        mytxt.append(file)

    EtestL,HtestL = [],[]
    # read content of files and store partitions IDs
    for f in mytxt:
        if f.startswith('train'):
            train = open(f, 'r')
            trainL = store_labels(train)
        elif f.startswith('val'):
            val = open(f, 'r')
            valL = store_labels(val)
        elif f.startswith('test'):
            test = open(f,'r')
            testL = store_labels(test)
        elif f.startswith('easy'):
            Etest = open(f,'r')
            EtestL = store_labels(Etest)
        elif f.startswith('hard'):
            Htest = open(f,'r')
            HtestL = store_labels(Htest)

    return data, trainL, valL, testL, EtestL, HtestL


def store_labels(label_file):
    # read each label file and store IDs
    storage_list = []
    content = label_file.readlines()
    for line in content:
        line1 = str(line.strip())
        f1 = str(line1.split(',')[0])
        if f1 != 'id':
            storage_list.append(str(f1))
    print(len(storage_list))
    return storage_list