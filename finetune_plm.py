import argparse
import random
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import BertForTokenClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_optimizer as custom_optim

from utils.bert_trainer import Trainer
from utils.bert_crf_trainer import CrfTrainer

from utils.data_loader import BertDataset
from utils.data_loader import TokenizerWrapper
from utils.models.bert_crf import BERT_CRF
from utils.models.bert_bilstm import BERT_BiLSTM
from utils.models.bert_bilstm_crf import BERT_BiLSTM_CRF


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_type', type=str, required=True)

    p.add_argument('--model_fn', required=True)

    p.add_argument('--train_fn', required=True)
    p.add_argument('--pretrained_model_name', type=str, default='bert-base-uncased')
    
    p.add_argument('--gpu_id', type=int, default=-1)

    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup_ratio', type=float, default=.1)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--use_radam', action='store_true')

    p.add_argument('--max_length', type=int, default=100)

    config = p.parse_args()

    return config


class SentenceGetter(object):

    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["word"].values.tolist(),
                                                           s['pos'].values.tolist(),
                                                           s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

def get_loaders(config, tokenizer):

    data = pd.read_csv(config.train_fn, sep='\t', encoding="utf-8")
    data.columns = ["sentence", "word", "pos", "tag"]
    getter = SentenceGetter(data)

    tag_lst = list(set(data["tag"].values))
    tag_lst.append("PAD")

    tag_to_index = {t : i for i, t in enumerate(tag_lst)}
    index_to_tag = {i : t for i, t in enumerate(tag_lst)}
    # index_to_tag = list(map(tag_to_index.get, tag_lst))
    print(tag_to_index)

    sentences = [[s[0] for s in sent] for sent in getter.sentences]
    labels = [[s[2] for s in sent] for sent in getter.sentences]

    # Shuffle before split into train and validation set.
    shuffled = list(zip(sentences, labels))
    random.shuffle(shuffled)
    sentences = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(sentences) * .8)  # train-validate ratio


    train_loader = DataLoader(
        BertDataset(sentences[:idx], labels[:idx]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TokenizerWrapper(tokenizer, config.max_length, tag_to_index).collate,
    )
    valid_loader = DataLoader(
        BertDataset(sentences[idx:], labels[idx:]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TokenizerWrapper(tokenizer, config.max_length, tag_to_index).collate,
    )

    return train_loader, valid_loader, index_to_tag

def get_optimizer(config, model):
    if config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.lr,
            eps=config.adam_epsilon
        )

    return optimizer

def get_scheduler(config, train_loader, optimizer):
    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )
    return scheduler

def main(config):

    # Get pretrained tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

    # Get dataloaders using tokenizer from untokenized corpus.
    train_loader, valid_loader, index_to_tag = get_loaders(config, tokenizer)

    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
    )


    if config.model_type == 'bert':
        model = BertForTokenClassification.from_pretrained(
            config.pretrained_model_name,
            num_labels=len(index_to_tag)
        )

        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, train_loader, optimizer)

        if config.gpu_id > -1 and torch.cuda.is_available():
            device = torch.device("cuda:{}".format(config.gpu_id))
            print('GPU on')
            print('Count of using GPUs:', torch.cuda.device_count())
        else:
            device = torch.device("cpu")
            print('No GPU')

        model.to(device)

        trainer = Trainer(config)
        model = trainer.train(
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader,
            index_to_tag,
            device,
        )

    elif config.model_type == 'bert_bilstm':
        model = BERT_BiLSTM(
            config,
            num_labels=len(index_to_tag),
        )

        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, train_loader, optimizer)

        if config.gpu_id > -1 and torch.cuda.is_available():
            device = torch.device("cuda:{}".format(config.gpu_id))
            print('GPU on')
            print('Count of using GPUs:', torch.cuda.device_count())
        else:
            device = torch.device("cpu")
            print('No GPU')

        model.to(device)

        trainer = Trainer(config)
        model = trainer.train(
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader,
            index_to_tag,
            device,
        )

    elif config.model_type == 'bert_crf':
        model = BERT_CRF(
            config,
            num_labels=len(index_to_tag),
        )

        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, train_loader, optimizer)

        if config.gpu_id > -1 and torch.cuda.is_available():
            device = torch.device("cuda:{}".format(config.gpu_id))
            print('GPU on')
            print('Count of using GPUs:', torch.cuda.device_count())
        else:
            device = torch.device("cpu")
            print('No GPU')

        model.to(device)

        trainer = CrfTrainer(config)
        model = trainer.train(
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader,
            index_to_tag,
            device,
        )

    elif config.model_type == 'bert_bilstm_crf':
        model = BERT_BiLSTM_CRF(
            config,
            num_labels=len(index_to_tag),
        )

        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, train_loader, optimizer)

        if config.gpu_id > -1 and torch.cuda.is_available():
            device = torch.device("cuda:{}".format(config.gpu_id))
            print('GPU on')
            print('Count of using GPUs:', torch.cuda.device_count())
        else:
            device = torch.device("cpu")
            print('No GPU')

        model.to(device)

        trainer = CrfTrainer(config)
        model = trainer.train(
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader,
            index_to_tag,
            device,
        )


    torch.save({
        'model': model.state_dict() if config.model_type is not None else None,
        'model_type': config.model_type if config.model_type is not None else None,
        'config': config,
        'vocab': None,
        'classes': index_to_tag,
        'tokenizer': tokenizer,
    }, config.model_fn)


if __name__ == '__main__':

    config = define_argparser()
    main(config)
