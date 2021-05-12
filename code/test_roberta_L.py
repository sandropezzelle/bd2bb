import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer

from datasets.BD2BBDataset import BD2BBDataset
from models.BD2BBRoBERTaModel import BD2BBRoBERTaModel
from utils import calculate_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_csv_filename", type=str, default="../data/train.csv")
    parser.add_argument("--val_dataset_csv_filename", type=str, default="../data/val.csv")
    parser.add_argument("--test_dataset_csv_filename", type=str, default="../data/test.csv")
    parser.add_argument("--filtering_dataset_csv_filename", type=str, default="../data/hard_unseen_test.csv")
    parser.add_argument("--whole_dataset_json_filename", type=str, default="../data/BD2BB.json")
    parser.add_argument(
        "--load_model_filename",
        type=str,
        default="../bin/roberta_L/roberta_run{}.model"
    )
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_runs", type=int, default=3)
    args = parser.parse_args()

    print("Loading dataset: {}".format(args.train_dataset_csv_filename))
    train_dataset = BD2BBDataset(
        split_dataset_csv_filename=args.train_dataset_csv_filename,
        whole_dataset_json_filename=args.whole_dataset_json_filename
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
        whole_dataset_json_filename=args.whole_dataset_json_filename
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
        whole_dataset_json_filename=args.whole_dataset_json_filename
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0
    )

    tokenizer = RobertaTokenizer.from_pretrained(
        "roberta-base",
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
                len(tokenizer.tokenize(sent1[i])) + len(tokenizer.tokenize(ending0[i])),
                len(tokenizer.tokenize(sent1[i])) + len(tokenizer.tokenize(ending1[i])),
                len(tokenizer.tokenize(sent1[i])) + len(tokenizer.tokenize(ending2[i])),
                len(tokenizer.tokenize(sent1[i])) + len(tokenizer.tokenize(ending3[i])),
                len(tokenizer.tokenize(sent1[i])) + len(tokenizer.tokenize(ending4[i]))
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
                len(tokenizer.tokenize(sent1[i])) + len(tokenizer.tokenize(ending0[i])),
                len(tokenizer.tokenize(sent1[i])) + len(tokenizer.tokenize(ending1[i])),
                len(tokenizer.tokenize(sent1[i])) + len(tokenizer.tokenize(ending2[i])),
                len(tokenizer.tokenize(sent1[i])) + len(tokenizer.tokenize(ending3[i])),
                len(tokenizer.tokenize(sent1[i])) + len(tokenizer.tokenize(ending4[i]))
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
                len(tokenizer.tokenize(sent1[i])) + len(tokenizer.tokenize(ending0[i])),
                len(tokenizer.tokenize(sent1[i])) + len(tokenizer.tokenize(ending1[i])),
                len(tokenizer.tokenize(sent1[i])) + len(tokenizer.tokenize(ending2[i])),
                len(tokenizer.tokenize(sent1[i])) + len(tokenizer.tokenize(ending3[i])),
                len(tokenizer.tokenize(sent1[i])) + len(tokenizer.tokenize(ending4[i]))
            )
    max_sentence_length += 4
    print("Maximum sentence length (including SEP and CLS tokens): {}".format(max_sentence_length))

    print("Loading filtering IDs from: {}".format(args.filtering_dataset_csv_filename))
    filter_ids = set()
    with open(args.filtering_dataset_csv_filename) as in_file:
        reader = csv.reader(in_file)
        next(reader)
        for row in reader:
            filter_ids.add(int(row[0].strip()))

    print("Loading dataset: {}".format(args.test_dataset_csv_filename))
    filtered_test_dataset = BD2BBDataset(
        split_dataset_csv_filename=args.test_dataset_csv_filename,
        whole_dataset_json_filename=args.whole_dataset_json_filename,
        filter_ids=filter_ids
    )
    print("Size of the filtered dataset: {}".format(len(filtered_test_dataset)))
    filtered_test_dataloader = DataLoader(
        dataset=filtered_test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0
    )

    runs_mean_filtered_test_accuracies = []
    for run in range(1, args.num_runs + 1):
        model = BD2BBRoBERTaModel(
            max_seq_length=max_sentence_length,
            from_scratch=True
        )
        model = model.cuda()
        print("Loading model: {}".format(args.load_model_filename.format(run)))
        model.load_state_dict(torch.load(args.load_model_filename.format(run)))
        model.eval()

        print("Computing mean accuracy on test set...")
        filtered_test_accuracies = []
        with torch.no_grad():
            for batch in tqdm(filtered_test_dataloader, total=len(filtered_test_dataloader), ncols=100):
                logits = model(
                    sent1=batch["sent1"],
                    ending0=batch["ending0"],
                    ending1=batch["ending1"],
                    ending2=batch["ending2"],
                    ending3=batch["ending3"],
                    ending4=batch["ending4"]
                )
                targets = batch["label"].cuda()
                filtered_test_accuracies.extend(calculate_accuracy(logits, targets))
            mean_filtered_test_accuracy = np.mean(filtered_test_accuracies)
            print("Mean filtered test accuracy (run {}): {}".format(run, mean_filtered_test_accuracy))
            runs_mean_filtered_test_accuracies.append(mean_filtered_test_accuracy)
    print("Mean filtered test accuracy: {}".format(np.mean(runs_mean_filtered_test_accuracies)))
    print("Std filtered test accuracy: {}".format(np.std(runs_mean_filtered_test_accuracies)))
