# https://github.com/eagle705/pytorch-bert-crf-ner

from transformers import BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF


class BERT_BiLSTM_CRF(nn.Module):
    def __init__(
            self,
            config,
            num_labels,
            lstm_hidden_size=256,
            n_layers=2,
            bert_dropout_p=0.1,
            bilstm_dropout_p=0.1
    ):
        # In BERTbase, the hidden size of last encoder's output of each token is 768
        self.input_size = 768
        self.num_labels = num_labels
        self.lstm_hidden_size = lstm_hidden_size
        self.n_layers = n_layers
        self.bert_dropout_p = bert_dropout_p
        self.bilstm_dropout_p = bilstm_dropout_p
        super(BERT_BiLSTM_CRF, self).__init__()

        self.bert = BertModel.from_pretrained(
            config.pretrained_model_name,
        )
        self.bert_dropout = nn.Dropout(bert_dropout_p)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.n_layers,
            dropout=self.bilstm_dropout_p,
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm_dropout = nn.Dropout(bilstm_dropout_p)
        self.generater = nn.Linear(self.lstm_hidden_size * 2, self.num_labels)
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        # self.activation = nn.LogSoftmax(dim=-1)
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
        # sequence_output = self.dropout(sequence_output)
        x = self.bert_dropout(sequence_output)
        # |x| = (batch_size, max_seq_length, 768)
        x, (h, c) = self.lstm(x)
        # |x| = (batch_size, max_seq_length, lstm_hidden_size * 2)
        x = self.bilstm_dropout(x)
        # |x| = (batch_size, max_seq_length, lstm_hidden_size * 2)
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


