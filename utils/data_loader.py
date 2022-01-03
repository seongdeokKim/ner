import torch
from torch.utils.data import Dataset
from keras.preprocessing.sequence import pad_sequences


class TokenizerWrapper():
    def __init__(self, tokenizer, max_length, tag_to_index):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tag_to_index = tag_to_index
        self.CLS, self.CLS_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.SEP, self.SEP_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.PAD, self.PAD_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id

    def collate(self, samples):
        sentences = [s['sentence'] for s in samples]
        labels = [s['label'] for s in samples]

        # Add a special token
        # and tokenize the text into subwords in a label-preserving way
        tokenized_sents_labels = [
            self.tokenize_and_preserve_labels(sent, labs)
            for sent, labs in zip(sentences, labels)
        ]

        sents = [[self.CLS] + sent_labs[0] + [self.SEP] for sent_labs in tokenized_sents_labels]
        labels = [["O"] + sent_labs[1] + ["O"] for sent_labs in tokenized_sents_labels]
        subwords_start_idxs = [[1.0] + sent_labs[2] + [1.0] for sent_labs in tokenized_sents_labels]

        # Convert subtokens to IDs
        input_ids = [self.tokenizer.convert_tokens_to_ids(subtoks) for subtoks in sents]
        # Convert sublables(subtags) to IDs
        labels = [[self.tag_to_index.get(l) for l in labs] for labs in labels]

        input_ids = pad_sequences(
            input_ids,
            maxlen=self.max_length,
            value=float(self.PAD_id),
            dtype="long",
            truncating="post",
            padding="post",
        )

        labels = pad_sequences(
            labels,
            maxlen=self.max_length,
            value=self.tag_to_index["PAD"],
            dtype="long",
            truncating="post",
            padding="post",
        )

        for voc_ids, lab_ids in zip(input_ids, labels):
            if voc_ids[-1] != self.PAD_id:
                voc_ids[-1], lab_ids[-1] = self.SEP_id, self.tag_to_index["O"]

        # Place a mask (zero) over the padding tokens
        attention_mask = [[float(i > 0) for i in ii] for ii in input_ids]

        # Place a mask (zero) over the subwords not in the first position of the word
        #attention_mask = self.use_subwords_for_training(attention_mask, subwords_start_idxs)

        return {
            'input_ids' : torch.tensor(input_ids, dtype=torch.long),
            'labels' : torch.tensor(labels, dtype=torch.long),
            'attention_mask' : torch.tensor(attention_mask, dtype=torch.long),
            'subwords_start_idxs': subwords_start_idxs
        }

    def tokenize_and_preserve_labels(self, sent, labs):
        """Segment each token into subwords and sublabels while keeping track of
        token boundaries

        Parameters
        ----------
        sent: A list of tokens regarding a sentence
        labs: A list of labels corresponding to each token

        Returns
        -------
        tokenized_sent: A list of subwords regarding a sentence
        labels : A list of sublabels corresponding to each subword
            if a word is divided into more than two,
            every sublabel belong to the word shares the orininal label
        subwords_start_idxs : A list of float(1) or float(0),
            which represents whether each subword is in first subword of the word
        """

        tokenized_sent = []
        labels = []
        subwords_start_idxs = []

        for word, lab in zip(sent, labs):
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_subwords = self.tokenizer.tokenize(str(word))
            n_subwords = len(tokenized_subwords)

            # Add the tokenized word to the final tokenized word list
            tokenized_sent.extend(tokenized_subwords)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([lab] * n_subwords)
            subwords_start_idxs.extend([1.0])
            subwords_start_idxs.extend([0.0] * (n_subwords-1))
            """
            if lab == "0":
                labels.extend([lab] * n_subwords)
                subwords_start_idxs.extend([1.0])
            else:
                labels.extend([lab] * n_subwords)
                subwords_start_idxs.extend([1.0])
                subwords_start_idxs.extend([0.0] * (n_subwords - 1))
            """
        return (tokenized_sent, labels, subwords_start_idxs)

    def use_subwords_for_training(self, attn_mask, subwords_start_idxs):
        # which means whether the other subwords are used or not for train & validate
        # Place a mask (zero) over the subwords of a word,
        # which are not in the first position
        for d1, (ii, _ii) in enumerate(zip(attn_mask, subwords_start_idxs)):
            for d2, (i, _i) in enumerate(zip(ii, _ii)):
                if _i == 0.0: attn_mask[d1][d2] = 0.0
        return attn_mask


class BertDataset(Dataset):
    '''This is version for BERT'''
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, item):
        sentence = self.sentences[item]
        label = self.labels[item]

        return {
            'sentence': sentence,
            'label': label,
        }
