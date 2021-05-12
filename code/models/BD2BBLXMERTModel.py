import numpy as np
import torch
import torch.nn as nn

from lxmert.src.lxrt.entry import InputFeatures
from lxmert.src.lxrt.modeling import LXRTFeatureExtraction as VisualBertForLXRFeature
from lxmert.src.lxrt.tokenization import BertTokenizer


def convert_bd2bb_sents_to_features(
        sents,
        ending0,
        ending1,
        ending2,
        ending3,
        ending4,
        max_seq_length,
        tokenizer,
        only_vision
):
    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())
        tokens_a_ending0 = tokenizer.tokenize(ending0[i].strip())
        tokens_a_ending1 = tokenizer.tokenize(ending1[i].strip())
        tokens_a_ending2 = tokenizer.tokenize(ending2[i].strip())
        tokens_a_ending3 = tokenizer.tokenize(ending3[i].strip())
        tokens_a_ending4 = tokenizer.tokenize(ending4[i].strip())

        def compute_ending_features(ending_tokens):
            tokens = ["[CLS]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)

        def compute_intention_ending_features(intention_tokens, ending_tokens):
            tokens = ["[CLS]"] + intention_tokens + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(intention_tokens) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)

        if only_vision:
            features.append(compute_ending_features(tokens_a_ending0))
            features.append(compute_ending_features(tokens_a_ending1))
            features.append(compute_ending_features(tokens_a_ending2))
            features.append(compute_ending_features(tokens_a_ending3))
            features.append(compute_ending_features(tokens_a_ending4))

        else:
            features.append(compute_intention_ending_features(tokens_a, tokens_a_ending0))
            features.append(compute_intention_ending_features(tokens_a, tokens_a_ending1))
            features.append(compute_intention_ending_features(tokens_a, tokens_a_ending2))
            features.append(compute_intention_ending_features(tokens_a, tokens_a_ending3))
            features.append(compute_intention_ending_features(tokens_a, tokens_a_ending4))

    return features


class BD2BBLXMERTModel(nn.Module):
    def __init__(self, lxrt_encoder_args, max_seq_length, hidden_dropout_prob=0.1, from_scratch=False,
                 only_vision=False):
        super(BD2BBLXMERTModel, self).__init__()

        self._max_seq_length = max_seq_length
        self._only_vision = only_vision

        self._tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        self._lxrt_encoder = VisualBertForLXRFeature.from_pretrained(
            "bert-base-uncased",
            mode="x"
        )
        if not from_scratch:
            self._load_pretrained_lxrt_encoder(lxrt_encoder_args.model_path)
        else:
            print("Initializing the model from scratch...")

        # self._classification_layer = nn.Sequential(
        #     nn.Linear(768, 768 * 2),
        #     GeLU(),
        #     BertLayerNorm(768 * 2, eps=1e-12),
        #     nn.Linear(768 * 2, 1)
        # )

        # Replicates RobertaForMultipleChoice. Indeed:
        # nn.Linear(768, 768) -> nn.Tanh() replicates the the pooled output
        # nn.Dropout(0.1) -> nn.Linear(768, 1) replicates the classification output
        self._classification_layer = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(768, 1)
        )

        self._classification_layer.apply(self._lxrt_encoder.init_bert_weights)

    def _load_pretrained_lxrt_encoder(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self._lxrt_encoder.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self._lxrt_encoder.load_state_dict(state_dict, strict=False)

    def forward(self, **kwargs):
        sent1 = kwargs["sent1"]
        ending0 = kwargs["ending0"]
        ending1 = kwargs["ending1"]
        ending2 = kwargs["ending2"]
        ending3 = kwargs["ending3"]
        ending4 = kwargs["ending4"]
        bottomup_features = kwargs["bottomup_features"]
        bottomup_boxes = kwargs["bottomup_boxes"]

        train_features = convert_bd2bb_sents_to_features(
            sent1,
            ending0,
            ending1,
            ending2,
            ending3,
            ending4,
            self._max_seq_length,
            self._tokenizer,
            self._only_vision
        )

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.float).cuda()

        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.cuda.LongTensor(
                np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
            return torch.index_select(a, dim, order_index)

        tiled_bottomup_features = tile(bottomup_features, 0, 5)
        tiled_bottomup_boxes = tile(bottomup_boxes, 0, 5)
        idx2token = {v: k for k, v in self._tokenizer.vocab.items()}
        encodings = self._lxrt_encoder(
            input_ids,
            attention_mask=input_mask,
            visual_feats=(tiled_bottomup_features, tiled_bottomup_boxes)
        )
        logits = self._classification_layer(encodings)
        reshaped_logits = logits.view(-1, 5)
        return reshaped_logits
