import csv
import json
import os

from torch.utils.data import Dataset


class BD2BBDataset(Dataset):
    def __init__(self, split_dataset_csv_filename, whole_dataset_json_filename, imgid2fasterRCNNfeatures=None,
                 filter_ids=None):
        super(BD2BBDataset, self).__init__()

        with open(whole_dataset_json_filename) as in_file:
            whole_dataset = json.load(in_file)

        self._dataset = {}
        _num_examples = 0
        with open(split_dataset_csv_filename) as in_file:
            reader = csv.DictReader(in_file)
            for row in reader:
                if filter_ids is not None and int(row["id"]) not in filter_ids:
                    continue

                self._dataset[_num_examples] = dict()
                self._dataset[_num_examples]["id"] = int(row["id"])
                self._dataset[_num_examples]["sent1"] = row["sent1"].lower()
                self._dataset[_num_examples]["sent2"] = row["sent2"].lower()
                self._dataset[_num_examples]["ending0"] = row["ending0"].lower()
                self._dataset[_num_examples]["ending1"] = row["ending1"].lower()
                self._dataset[_num_examples]["ending2"] = row["ending2"].lower()
                self._dataset[_num_examples]["ending3"] = row["ending3"].lower()
                self._dataset[_num_examples]["ending4"] = row["ending4"].lower()
                self._dataset[_num_examples]["end0type"] = row["end0type"]
                self._dataset[_num_examples]["end1type"] = row["end1type"]
                self._dataset[_num_examples]["end2type"] = row["end2type"]
                self._dataset[_num_examples]["end3type"] = row["end3type"]
                self._dataset[_num_examples]["end4type"] = row["end4type"]
                self._dataset[_num_examples]["label"] = int(row["label"])
                self._dataset[_num_examples]["image_url"] = whole_dataset[row["id"]][0]["target_item"]["image_url"]
                _num_examples += 1

        if imgid2fasterRCNNfeatures is not None:
            print("Found {} examples before taking images into account".format(len(self._dataset)))

            filtered_dataset = {}
            _num_examples = 0
            for k, example in self._dataset.items():
                image_id = os.path.splitext(os.path.basename(os.path.normpath(self._dataset[k]["image_url"])))[0]
                if image_id in imgid2fasterRCNNfeatures:
                    filtered_dataset[_num_examples] = example
                    filtered_dataset[_num_examples]["bottomup_features"] = imgid2fasterRCNNfeatures[image_id][
                        "features"]
                    filtered_dataset[_num_examples]["bottomup_unnorm_boxes"] = imgid2fasterRCNNfeatures[image_id][
                        "boxes"]
                    boxes = imgid2fasterRCNNfeatures[image_id]["boxes"].clone()
                    boxes[:, (0, 2)] /= imgid2fasterRCNNfeatures[image_id]["img_w"]
                    boxes[:, (1, 3)] /= imgid2fasterRCNNfeatures[image_id]["img_h"]
                    filtered_dataset[_num_examples]["bottomup_boxes"] = boxes
                    _num_examples += 1
            self._dataset = filtered_dataset

            print("Found {} examples after taking images into account".format(len(self._dataset)))

        else:
            print("Found {} examples".format(len(self._dataset)))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]
