import argparse
import csv
import json
from types import SimpleNamespace

import numpy as np
import sharearray
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.BD2BBDataset import BD2BBDataset
from lxmert.src.lxrt.optimization import BertAdam
from lxmert.src.lxrt.tokenization import BertTokenizer
from models.BD2BBLXMERTModel import BD2BBLXMERTModel
from utils import calculate_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_csv_filename", type=str, default="../data/train.csv")
    parser.add_argument("--val_dataset_csv_filename", type=str, default="../data/val.csv")
    parser.add_argument("--test_dataset_csv_filename", type=str, default="../data/test.csv")
    parser.add_argument("--hard_unseen_test_dataset_csv_filename", type=str, default="../data/hard_unseen_test.csv")

    parser.add_argument("--whole_dataset_json_filename", type=str, default="../data/BD2BB.json")
    parser.add_argument(
        "--load_mscoco_bottomup_info_json_filename",
        type=str,
        default="../data/mscoco_features/mscoco_bottomup_info.json"
    )
    parser.add_argument(
        "--load_mscoco_bottomup_features_npy_filename",
        type=str,
        default="../data/mscoco_features/mscoco_bottomup_features.npy"
    )
    parser.add_argument(
        "--load_mscoco_bottomup_boxes_npy_filename",
        type=str,
        default="../data/mscoco_features/mscoco_bottomup_boxes.npy"
    )
    parser.add_argument(
        "--save_best_model_path",
        type=str,
        default="../bin/lxmert_V/lxmert.model"
    )
    parser.add_argument(
        "--save_log_path",
        type=str,
        default="../bin/lxmert_V/lxmert.log"
    )
    parser.add_argument(
        "--save_args_path",
        type=str,
        default="../bin/lxmert_V/lxmert.args"
    )
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--grad_acc_steps", type=int, default=8)
    parser.add_argument("--from_scratch", type=bool, default=False, action="store_true")
    args = parser.parse_args()

    print("Writing args to: {}".format(args.save_args_path))
    with open(args.save_args_path, mode="w") as out_file:
        json.dump(vars(args), out_file)

    print("Loading MS-COCO Bottom-Up info from: {}".format(args.load_mscoco_bottomup_info_json_filename))
    with open(args.load_mscoco_bottomup_info_json_filename) as in_file:
        mscoco_bottomup_index = json.load(in_file)
        image_id2image_pos = mscoco_bottomup_index["image_id2image_pos"]
        image_pos2image_id = mscoco_bottomup_index["image_pos2image_id"]
        img_h = mscoco_bottomup_index["img_h"]
        img_w = mscoco_bottomup_index["img_w"]

    print("Loading MS-COCO Bottom-Up features from: {}".format(args.load_mscoco_bottomup_features_npy_filename))
    mscoco_bottomup_features = torch.from_numpy(np.load(args.load_mscoco_bottomup_features_npy_filename))

    print("Loading MS-COCO Bottom-Up boxes from: {}".format(args.load_mscoco_bottomup_boxes_npy_filename))
    mscoco_bottomup_boxes = torch.from_numpy(np.load(args.load_mscoco_bottomup_boxes_npy_filename))

    imgid2fasterRCNNfeatures = {}
    for mscoco_id, mscoco_pos in image_id2image_pos.items():
        imgid2fasterRCNNfeatures[mscoco_id] = dict()
        imgid2fasterRCNNfeatures[mscoco_id]["features"] = mscoco_bottomup_features[mscoco_pos]
        imgid2fasterRCNNfeatures[mscoco_id]["boxes"] = mscoco_bottomup_boxes[mscoco_pos]
        imgid2fasterRCNNfeatures[mscoco_id]["img_h"] = img_h[mscoco_pos]
        imgid2fasterRCNNfeatures[mscoco_id]["img_w"] = img_w[mscoco_pos]

    print("Loading dataset: {}".format(args.train_dataset_csv_filename))
    train_dataset = BD2BBDataset(
        split_dataset_csv_filename=args.train_dataset_csv_filename,
        whole_dataset_json_filename=args.whole_dataset_json_filename,
        imgid2fasterRCNNfeatures=imgid2fasterRCNNfeatures
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=0
    )

    print("Loading dataset: {}".format(args.val_dataset_csv_filename))
    val_dataset = BD2BBDataset(
        split_dataset_csv_filename=args.val_dataset_csv_filename,
        whole_dataset_json_filename=args.whole_dataset_json_filename,
        imgid2fasterRCNNfeatures=imgid2fasterRCNNfeatures
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0
    )

    print("Loading dataset: {}".format(args.test_dataset_csv_filename))
    test_dataset = BD2BBDataset(
        split_dataset_csv_filename=args.test_dataset_csv_filename,
        whole_dataset_json_filename=args.whole_dataset_json_filename,
        imgid2fasterRCNNfeatures=imgid2fasterRCNNfeatures
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0
    )

    if args.hard_unseen_test_dataset_csv_filename is not None:
        print("Loading dataset: {}".format(args.hard_unseen_test_dataset_csv_filename))
        hard_unseen_test_dataset = BD2BBDataset(
            split_dataset_csv_filename=args.hard_unseen_test_dataset_csv_filename,
            whole_dataset_json_filename=args.whole_dataset_json_filename,
            imgid2fasterRCNNfeatures=imgid2fasterRCNNfeatures
        )
        hard_unseen_test_dataloader = DataLoader(
            dataset=hard_unseen_test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=0
        )

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True
    )
    print("Computing maximum training sentence length...")
    max_sentence_length = 0
    for batch in train_dataloader:
        sent1 = batch["sent1"]
        ending0 = batch["ending0"]
        ending1 = batch["ending1"]
        ending2 = batch["ending2"]
        ending3 = batch["ending3"]
        ending4 = batch["ending4"]
        for i in range(len(batch["sent1"])):
            max_sentence_length = max(
                max_sentence_length,
                len(tokenizer.tokenize(ending0[i])),
                len(tokenizer.tokenize(ending1[i])),
                len(tokenizer.tokenize(ending2[i])),
                len(tokenizer.tokenize(ending3[i])),
                len(tokenizer.tokenize(ending4[i]))
            )
    for batch in val_dataloader:
        sent1 = batch["sent1"]
        ending0 = batch["ending0"]
        ending1 = batch["ending1"]
        ending2 = batch["ending2"]
        ending3 = batch["ending3"]
        ending4 = batch["ending4"]
        for i in range(len(batch["sent1"])):
            max_sentence_length = max(
                max_sentence_length,
                len(tokenizer.tokenize(ending0[i])),
                len(tokenizer.tokenize(ending1[i])),
                len(tokenizer.tokenize(ending2[i])),
                len(tokenizer.tokenize(ending3[i])),
                len(tokenizer.tokenize(ending4[i]))
            )
    for batch in test_dataloader:
        sent1 = batch["sent1"]
        ending0 = batch["ending0"]
        ending1 = batch["ending1"]
        ending2 = batch["ending2"]
        ending3 = batch["ending3"]
        ending4 = batch["ending4"]
        for i in range(len(batch["sent1"])):
            max_sentence_length = max(
                max_sentence_length,
                len(tokenizer.tokenize(ending0[i])),
                len(tokenizer.tokenize(ending1[i])),
                len(tokenizer.tokenize(ending2[i])),
                len(tokenizer.tokenize(ending3[i])),
                len(tokenizer.tokenize(ending4[i]))
            )
    if args.hard_unseen_test_dataset_csv_filename is not None:
        for batch in hard_unseen_test_dataloader:
            sent1 = batch["sent1"]
            ending0 = batch["ending0"]
            ending1 = batch["ending1"]
            ending2 = batch["ending2"]
            ending3 = batch["ending3"]
            ending4 = batch["ending4"]
            for i in range(len(batch["sent1"])):
                max_sentence_length = max(
                    max_sentence_length,
                    len(tokenizer.tokenize(ending0[i])),
                    len(tokenizer.tokenize(ending1[i])),
                    len(tokenizer.tokenize(ending2[i])),
                    len(tokenizer.tokenize(ending3[i])),
                    len(tokenizer.tokenize(ending4[i]))
                )
    max_sentence_length += 3
    print("Maximum sentence length (including SEP and CLS tokens): {}".format(max_sentence_length))

    model = BD2BBLXMERTModel(
        SimpleNamespace(
            **{
                "llayers": 9,
                "xlayers": 5,
                "rlayers": 5,
                "model_path": "/mnt/8tera/claudio.greco/guesswhat_lxmert/guesswhat/lxmert/snap/pretrained/model"
            }
        ),
        max_seq_length=max_sentence_length,
        from_scratch=args.from_scratch,
        only_vision=True
    )
    model = model.cuda()

    batch_per_epoch = len(train_dataloader)
    num_total_batches = int(batch_per_epoch * args.num_epochs)
    optimizer = BertAdam(
        list(model.parameters()),
        warmup=0.1,
        lr=args.learning_rate,
        t_total=num_total_batches
    )

    loss_function = nn.CrossEntropyLoss()

    epoch_train_mean_accuracies = []
    epoch_val_mean_accuracies = []
    epoch_test_mean_accuracies = []
    epoch_hard_unseen_test_mean_accuracies = []
    best_validation_accuracy = 0

    for num_epoch in range(args.num_epochs):
        print("\nStarting epoch {}".format(num_epoch))

        print("Training...")
        model.train()
        stream = tqdm(train_dataloader, total=len(train_dataloader), ncols=100)
        for batch_index, batch in enumerate(stream):
            logits = model(
                sent1=batch["sent1"],
                ending0=batch["ending0"],
                ending1=batch["ending1"],
                ending2=batch["ending2"],
                ending3=batch["ending3"],
                ending4=batch["ending4"],
                bottomup_features=batch["bottomup_features"].cuda(),
                bottomup_boxes=batch["bottomup_boxes"].cuda()
            )

            # optimizer.zero_grad()
            targets = batch["label"].cuda()
            loss = loss_function(logits, targets)
            stream.set_description("Loss: {}".format(loss))
            loss = loss / args.grad_acc_steps
            loss.backward()

            if (batch_index + 1) % args.grad_acc_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()

        print("Computing mean accuracy on training set...")
        model.eval()
        train_accuracies = []
        with torch.no_grad():
            for batch in tqdm(train_dataloader, total=len(train_dataloader), ncols=100):
                logits = model(
                    sent1=batch["sent1"],
                    ending0=batch["ending0"],
                    ending1=batch["ending1"],
                    ending2=batch["ending2"],
                    ending3=batch["ending3"],
                    ending4=batch["ending4"],
                    bottomup_features=batch["bottomup_features"].cuda(),
                    bottomup_boxes=batch["bottomup_boxes"].cuda()
                )
                targets = batch["label"].cuda()
                train_accuracies.extend(calculate_accuracy(logits, targets))
            mean_train_accuracy = np.mean(train_accuracies)
            epoch_train_mean_accuracies.append(mean_train_accuracy)
            print("Mean training accuracy: {}".format(mean_train_accuracy))

        print("Computing mean accuracy on validation set...")
        model.eval()
        val_accuracies = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, total=len(val_dataloader), ncols=100):
                logits = model(
                    sent1=batch["sent1"],
                    ending0=batch["ending0"],
                    ending1=batch["ending1"],
                    ending2=batch["ending2"],
                    ending3=batch["ending3"],
                    ending4=batch["ending4"],
                    bottomup_features=batch["bottomup_features"].cuda(),
                    bottomup_boxes=batch["bottomup_boxes"].cuda()
                )
                targets = batch["label"].cuda()
                val_accuracies.extend(calculate_accuracy(logits, targets))
            mean_val_accuracy = np.mean(val_accuracies)
            epoch_val_mean_accuracies.append(mean_val_accuracy)
            print("Mean validation accuracy: {}".format(mean_val_accuracy))

        if mean_val_accuracy > best_validation_accuracy:
            print("Obtained best model at epoch: {}".format(num_epoch))
            best_validation_accuracy = mean_val_accuracy

            print("Saving model to: {}".format(args.save_best_model_path))
            torch.save(model.state_dict(), args.save_best_model_path)

        print("Computing mean accuracy on test set...")
        test_accuracies = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader, total=len(test_dataloader), ncols=100):
                logits = model(
                    sent1=batch["sent1"],
                    ending0=batch["ending0"],
                    ending1=batch["ending1"],
                    ending2=batch["ending2"],
                    ending3=batch["ending3"],
                    ending4=batch["ending4"],
                    bottomup_features=batch["bottomup_features"].cuda(),
                    bottomup_boxes=batch["bottomup_boxes"].cuda()
                )
                targets = batch["label"].cuda()
                test_accuracies.extend(calculate_accuracy(logits, targets))
            mean_test_accuracy = np.mean(test_accuracies)
            epoch_test_mean_accuracies.append(mean_test_accuracy)
            print("Mean test accuracy: {}".format(mean_test_accuracy))

        if args.hard_unseen_test_dataset_csv_filename is not None:
            print("Computing mean accuracy on hard unseen test set...")
            hard_unseen_test_accuracies = []
            with torch.no_grad():
                for batch in tqdm(hard_unseen_test_dataloader, total=len(hard_unseen_test_dataloader), ncols=100):
                    logits = model(
                        sent1=batch["sent1"],
                        ending0=batch["ending0"],
                        ending1=batch["ending1"],
                        ending2=batch["ending2"],
                        ending3=batch["ending3"],
                        ending4=batch["ending4"],
                        bottomup_features=batch["bottomup_features"].cuda(),
                        bottomup_boxes=batch["bottomup_boxes"].cuda()
                    )
                    targets = batch["label"].cuda()
                    hard_unseen_test_accuracies.extend(calculate_accuracy(logits, targets))
                mean_hard_unseen_test_accuracy = np.mean(hard_unseen_test_accuracies)
                epoch_hard_unseen_test_mean_accuracies.append(mean_hard_unseen_test_accuracy)
                print("Mean hard unseen test accuracy: {}".format(mean_hard_unseen_test_accuracy))

    if args.hard_unseen_test_dataset_csv_filename is not None:
        print("Writing log to: {}".format(args.save_log_path))
        with open(args.save_log_path, mode="w") as out_file:
            writer = csv.writer(out_file)
            writer.writerow(["Epoch", "Training", "Validation", "Test", "Hard Unseen Test"])
            for i in range(args.num_epochs):
                writer.writerow(
                    [
                        i,
                        epoch_train_mean_accuracies[i],
                        epoch_val_mean_accuracies[i],
                        epoch_test_mean_accuracies[i],
                        epoch_hard_unseen_test_mean_accuracies[i]
                    ]
                )
    else:
        print("Writing log to: {}".format(args.save_log_path))
        with open(args.save_log_path, mode="w") as out_file:
            writer = csv.writer(out_file)
            writer.writerow(["Epoch", "Training", "Validation", "Test"])
            for i in range(args.num_epochs):
                writer.writerow(
                    [
                        i,
                        epoch_train_mean_accuracies[i],
                        epoch_val_mean_accuracies[i],
                        epoch_test_mean_accuracies[i]
                    ]
                )
