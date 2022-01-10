from transformers import BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

class BERT_BiLSTM(nn.Module):
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
        super(BERT_BiLSTM, self).__init__()

        self.bert = BertModel.from_pretrained(
            config.pretrained_model_name,
        )
        self.bert_dropout = nn.Dropout(bert_dropout_p)
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
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
        #self.activation = nn.LogSoftmax(dim=-1)

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
        x, (h, c) = self.lstm(x)  ## extract the 1st token's embeddings
        # |x| = (batch_size, max_seq_length, lstm_hidden_size * 2)
        x = self.bilstm_dropout(x)
        # |x| = (batch_size, max_seq_length, lstm_hidden_size * 2)
        logits = self.generater(x)
        # |logits| = (batch_size, max_seq_length, num_labels)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output