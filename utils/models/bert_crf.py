# https://github.com/eagle705/pytorch-bert-crf-ner

from transformers import BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF


class BERT_CRF(nn.Module):
    def __init__(
            self,
            config,
            num_labels,
            bert_dropout_p=0.1,
    ):
        # In BERTbase, the hidden size of last encoder's output of each token is 768
        self.input_size = 768
        self.num_labels = num_labels
        self.bert_dropout_p = bert_dropout_p
        super(BERT_CRF, self).__init__()

        self.bert = BertModel.from_pretrained(
            config.pretrained_model_name,
        )
        self.bert_dropout = nn.Dropout(bert_dropout_p)
        self.generater = nn.Linear(768, self.num_labels)
        self.crf = CRF(
            num_tags=self.num_labels,
            batch_first=True,
        )

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            return_dict=None,
    ):

        sequence_output, pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        # |sequence_output| = (batch_size, max_seq_length, 768)
        # |pooled_output| = (batch_size, 768)
        x = self.bert_dropout(sequence_output)
        # |x| = (batch_size, max_seq_length, 768)
        emissions = self.generater(x)
        # |emissions| = (batch_size, max_seq_length, num_labels)
        sequence_of_tags = self.crf.decode(emissions)
        # |sequence_of_tags| = (batch_size, max_seq_length)
        sequence_of_tags = torch.tensor(sequence_of_tags)

        loss = None
        if labels is not None:
            log_likelihood = self.crf(emissions, labels, reduction='mean')
            # loss is a negative log-likelihood
            loss = -1 * log_likelihood

        if not return_dict:
            output = (sequence_of_tags,)
            return ((loss,) + output) if loss is not None else output