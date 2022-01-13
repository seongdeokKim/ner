import torch
from torch.utils.data import Dataset
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize


class BertDataset(Dataset):

    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        label = self.labels[item]

        return sentence, label


class TokenizerWrapper():
    def __init__(self, tokenizer,
                 max_length,
                 tag_to_index,
                 reflect_only_first_bpe_token_during_training=None):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tag_to_index = tag_to_index

        self.CLS, self.CLS_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.SEP, self.SEP_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.PAD, self.PAD_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id

        self.reflect_only_first_bpe_token_during_training = reflect_only_first_bpe_token_during_training

    def generate_for_train(self, samples: 'list of tuple(sentence, label)'):

        sentences = [s[0] for s in samples]
        labels = [s[1] for s in samples]

        # Add a special token
        # and tokenize the text into subwords in a label-preserving way
        bpe_based_features = [
            self.transform_bio_to_bpe(bio_tokens_per_sent, bio_lables_per_sent)
            for bio_tokens_per_sent, bio_lables_per_sent in zip(sentences, labels)
        ]

        bpe_labels = [["O"] + t[0] + ["O"] for t in bpe_based_features]
        bpe_tokens = [[self.CLS] + t[1] + [self.SEP] for t in bpe_based_features]
        bpe_token_start_idxs = [[1.0] + t[2] + [1.0] for t in bpe_based_features]

        # Convert tokens to token_ids
        input_ids = [
            self.tokenizer.convert_tokens_to_ids(bpe_tokens_per_sent)
            for bpe_tokens_per_sent in bpe_tokens
        ]

        # Convert lables to label_ids
        labels = [
            [self.tag_to_index.get(bpe_label) for bpe_label in bpe_labels_per_sent]
            for bpe_labels_per_sent in bpe_labels
        ]

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

        for input_ids_per_sent, labels_per_sent in zip(input_ids, labels):
            if input_ids_per_sent[-1] != self.PAD_id:
                input_ids_per_sent[-1] = self.SEP_id
                labels_per_sent[-1] = self.tag_to_index["O"]

        # Place a mask (zero) over the padding tokens
        attention_mask = [
            [float(input_id > 0) for input_id in input_ids_per_sent]
            for input_ids_per_sent in input_ids
        ]

        bpe_token_start_idxs = pad_sequences(
            bpe_token_start_idxs,
            maxlen=self.max_length,
            value=float(0.0),
            # dtype="long",
            truncating="post",
            padding="post",
        )

        # Place a mask (zero) over the follwer tokens
        # Follower tokens means tokens that follows the token in the first position of each CoNLL token
        if self.reflect_only_first_bpe_token_during_training is not None:
            attention_mask = self.update_attention_mask(attention_mask,
                                                        bpe_token_start_idxs)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'bpe_token_start_idxs': bpe_token_start_idxs
        }

    def generate_for_predict(self, samples: 'list of str(sentence)'):

        sentences = [word_tokenize(s) for s in samples]

        bert_based_features = [
            self.transform_bio_to_bpe(bio_tokens_per_sent)
            for bio_tokens_per_sent in sentences
        ]

        bpe_tokens = [[self.CLS] + t[0] + [self.SEP] for t in bert_based_features]
        bpe_token_start_idxs = [[1.0] + t[1] + [1.0] for t in bert_based_features]

        # Convert tokens to token_ids and padding
        input_ids = [
            self.tokenizer.convert_tokens_to_ids(bpe_tokens_per_sent)
            for bpe_tokens_per_sent in bpe_tokens
        ]

        input_ids = pad_sequences(
            input_ids,
            maxlen=self.max_length,
            value=float(self.PAD_id),
            dtype="long",
            truncating="post",
            padding="post",
        )

        # Swaps out the final token for [SEP]
        # for any sequences whose length is over the MAX_LEN
        for input_ids_per_sent in input_ids:
            if input_ids_per_sent[-1] != self.PAD_id:
                input_ids_per_sent[-1] = self.SEP_id

        # Place a mask (zero) over the padding tokens
        attention_mask = [
            [float(input_id > 0) for input_id in input_ids_per_sent]
            for input_ids_per_sent in input_ids
        ]

        # padding bpe_token_start_idxs
        bpe_token_start_idxs = pad_sequences(
            bpe_token_start_idxs,
            maxlen=self.max_length,
            value=float(0.0),
            # dtype="long",
            truncating="post",
            padding="post",
        )

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'bpe_token_start_idxs': bpe_token_start_idxs
        }

    def transform_bio_to_bpe(self, bio_tokens_per_sent,
                             bio_lables_per_sent=None):
        """Segment each of bio algorithm based token into BPE based tokens and labels

        Parameters
        ----------
        bio_tokens_per_sent: A list of tokens (bio based) regarding a sentence
        bio_lables_per_sent: A list of labels (bio based) corresponding to each token

        Returns
        -------
        bpe_tokens_per_sent: A list of tokens(BPE based) regarding a sentence
        bpe_token_start_idxs _per_sent: A list of float(1) or float(0),
            which represents whether each bpe_token is the first subword of CoNLL token
        bpe_labels_per_sent : A list of labels(BPE based) corresponding to each token
            if a word is divided into more than two,
            every sub label belong to the word shares the original label
        """

        bpe_tokens_per_sent = []
        bpe_token_start_idxs_per_sent = []
        bpe_labels_per_sent = []

        for i in range(len(bio_tokens_per_sent)):
            bio_token = bio_tokens_per_sent[i]
            bpe_tokens = self.tokenizer.tokenize(str(bio_token))

            bpe_length = len(bpe_tokens)
            bpe_tokens_per_sent += bpe_tokens

            # Add the same label to all bert tokens that share same conll token
            if bio_lables_per_sent is not None:
                bio_lable = bio_lables_per_sent[i]
                bpe_labels_per_sent += ([bio_lable] * bpe_length)

            # Check whether each bert token is the first position of corresponding one conll token
            bpe_token_start_idxs_per_sent += [1.0]
            bpe_token_start_idxs_per_sent += ([0.0] * (bpe_length - 1))

            output = (bpe_tokens_per_sent, bpe_token_start_idxs_per_sent,)
            if bio_lables_per_sent is not None:
                output = (bpe_labels_per_sent,) + output

        return output

    @staticmethod
    def update_attention_mask(attention_mask,
                              bpe_token_start_idxs):

        # bpe tokens, which are not in the first position, could be used or not
        # for fine-tuning phase
        # If you want to reflect only first position, you should update attention mask
        for i in range(len(attention_mask)):
            for j in range(len(attention_mask[i])):
                if float(bpe_token_start_idxs[i][j]) == float(0.0):
                    attention_mask[i][j] = float(0.0)

        return attention_mask
