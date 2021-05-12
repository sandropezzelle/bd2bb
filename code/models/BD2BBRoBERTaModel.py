import torch
import torch.nn as nn
from transformers import RobertaModel

from transformers import RobertaTokenizer


class InputFeatures(object):
    def __init__(self, input_ids, input_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask


def convert_bd2bb_sents_to_features(
        sents,
        ending0,
        ending1,
        ending2,
        ending3,
        ending4,
        max_seq_length,
        tokenizer,
        without_intention
):
    features = []
    for i, sent in enumerate(sents):
        def compute_ending_features(ending):
            encoding = tokenizer.encode_plus(
                ending.strip().lower(),
                add_special_tokens=True,
                max_length=max_seq_length
            )

            input_ids = encoding["input_ids"]
            input_mask = encoding["attention_mask"]

            # Here padding is equal to 1 because RoBERTa has config.pad_token_id = 1
            padding = [1] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length

            return InputFeatures(input_ids=input_ids, input_mask=input_mask)

        def compute_intention_ending_features(intention, ending):
            encoding = tokenizer.encode_plus(
                intention.strip().lower(),
                ending.strip().lower(),
                add_special_tokens=True,
                max_length=max_seq_length
            )

            input_ids = encoding["input_ids"]
            input_mask = encoding["attention_mask"]

            # Here padding is equal to 1 because RoBERTa has config.pad_token_id = 1
            padding = [1] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length

            return InputFeatures(input_ids=input_ids, input_mask=input_mask)

        if without_intention:
            features.append(compute_ending_features(ending0[i]))
            features.append(compute_ending_features(ending1[i]))
            features.append(compute_ending_features(ending2[i]))
            features.append(compute_ending_features(ending3[i]))
            features.append(compute_ending_features(ending4[i]))

        else:
            features.append(compute_intention_ending_features(sent, ending0[i]))
            features.append(compute_intention_ending_features(sent, ending1[i]))
            features.append(compute_intention_ending_features(sent, ending2[i]))
            features.append(compute_intention_ending_features(sent, ending3[i]))
            features.append(compute_intention_ending_features(sent, ending4[i]))

    return features


class BD2BBRoBERTaModel(nn.Module):
    def __init__(self, max_seq_length, hidden_dropout_prob=0.1, from_scratch=False, without_intention=False):
        super(BD2BBRoBERTaModel, self).__init__()

        self._max_seq_length = max_seq_length
        self._without_intention = without_intention
        self._tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
        self._roberta = RobertaModel.from_pretrained("roberta-base")
        if from_scratch:
            print("Initializing the model from scratch...")
            self._roberta = RobertaModel(self._roberta.config)
        self._classification_layer = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(768, 1)
        )

    def forward(self, **kwargs):
        sent1 = kwargs["sent1"]
        ending0 = kwargs["ending0"]
        ending1 = kwargs["ending1"]
        ending2 = kwargs["ending2"]
        ending3 = kwargs["ending3"]
        ending4 = kwargs["ending4"]

        features = convert_bd2bb_sents_to_features(
            sent1,
            ending0,
            ending1,
            ending2,
            ending3,
            ending4,
            self._max_seq_length,
            self._tokenizer,
            self._without_intention
        )

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.float).cuda()

        outputs = self._roberta(input_ids=input_ids, attention_mask=input_mask)
        pooled_output = outputs[1]
        logits = self._classification_layer(pooled_output)
        reshaped_logits = logits.view(-1, 5)
        return reshaped_logits
