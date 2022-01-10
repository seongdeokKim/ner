import torch
from torch.utils.data import Dataset
from keras.preprocessing.sequence import pad_sequences
import re
from nltk.tokenize import word_tokenize


##################################################################################
###############################  TRAINING  ######################################
##################################################################################

class BertDataset(Dataset):

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

    def collate(self, samples):

        sentences = [s['sentence'] for s in samples]
        labels = [s['label'] for s in samples]

        # Add a special token
        # and tokenize the text into subwords in a label-preserving way
        bpe_based_features = [
            self.transform_bio_to_bpe(bio_tokens_per_sent, bio_lables_per_sent)
            for bio_tokens_per_sent, bio_lables_per_sent in zip(sentences, labels)
        ]

        bpe_tokens = [[self.CLS] + t[0] + [self.SEP] for t in bpe_based_features]
        bpe_labels = [["O"] + t[1] + ["O"] for t in bpe_based_features]
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
            #dtype="long",
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

    def transform_bio_to_bpe(self, bio_tokens_per_sent,
                              bio_lables_per_sent):
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

        for bio_token, bio_lable in zip(bio_tokens_per_sent, bio_lables_per_sent):
            # Subtokenize each bio token into bpe token(s) and count the number of bpe token(s)
            bpe_tokens = self.tokenizer.tokenize(str(bio_token))

            bpe_length = len(bpe_tokens)
            bpe_tokens_per_sent += bpe_tokens

            # Add the same label to all bert tokens that share same conll token
            bpe_labels_per_sent += ([bio_lable] * bpe_length)

            # Check whether each bert token is the first position of corresponding one conll token
            bpe_token_start_idxs_per_sent += [1.0]
            bpe_token_start_idxs_per_sent += ([0.0] * (bpe_length-1))

        return bpe_tokens_per_sent, bpe_labels_per_sent, bpe_token_start_idxs_per_sent

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



##################################################################################
###############################  INFERENCE  ######################################
##################################################################################

class TokenizerForInference:
    def __init__(self, tokenizer,
                 max_length,
                 index_to_tag):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.index_to_tag = index_to_tag

        self.CLS, self.CLS_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.SEP, self.SEP_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.PAD, self.PAD_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id


    def tokenize(self, lines):

        sentences = [word_tokenize(line) for line in lines]

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
            #dtype="long",
            truncating="post",
            padding="post",
        )

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'bpe_token_start_idxs': bpe_token_start_idxs
        }

    def transform_bio_to_bpe(self, bio_tokens_per_sent):
        """Segment each of CoNLL based token into BPE based tokens and labels corresponding to each bpe based token

        Parameters
        ----------
        bio_tokens_per_sent:

        Returns
        -------
        bpe_tokens_per_sent: A list of tokens(BPE based) regarding a sentence
        bpe_token_start_idxs_per_sent : A list of which value is float(1.0) or float(0.0),
            which represents whether each bpe_token is the first subtoken of one bio token
        """

        bpe_tokens_per_sent = []
        bpe_token_start_idxs_per_sent = []

        for bio_token in bio_tokens_per_sent:

            # Subtokenize each conll token into bert token(s) and count the number of subwords
            bpe_tokens = self.tokenizer.tokenize(str(bio_token))

            bpe_length = len(bpe_tokens)
            bpe_tokens_per_sent += bpe_tokens

            # Check whether each bert token is the first position of corresponding one conll token
            bpe_token_start_idxs_per_sent += [1.0]
            bpe_token_start_idxs_per_sent += ([0.0] * (bpe_length-1))

        return bpe_tokens_per_sent, bpe_token_start_idxs_per_sent


class PipelineForBPEtoNGAM:

    def __init__(self, tokenizer,
                 index_to_tag):

        self.tokenizer = tokenizer
        self.index_to_tag = index_to_tag
        self.CLS, self.CLS_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.SEP, self.SEP_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.PAD, self.PAD_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id

    def run(self, preds,
            bpe_token_ids,
            bpe_token_start_idxs):

        # From bpe format to bio, and to ngram.
        pred_tags, bpe_tokens = self.convert_id_to_bpe(preds, bpe_token_ids)
        bio_pred_tags, bio_tokens = self.transform_bpe_to_bio(pred_tags, bpe_tokens, bpe_token_start_idxs)
        ngram_pred_tags, ngram_tokens = self.transform_bio_to_ngram(bio_pred_tags, bio_tokens)

        # preprocess previously sentences
        sents = self.preprocess_sentence(bio_tokens)

        # get span info. of ngram token that are extracted above process
        ngram_pred_tags, ngram_tokens, ngram_spans = self.get_span_info(ngram_pred_tags, ngram_tokens, sents)

        return {
            "ngram_pred_tags": ngram_pred_tags,
            "ngram_tokens": ngram_tokens,
            "ngram_spans": ngram_spans,
            "preprocessed_sents": sents
        }

    def convert_id_to_bpe(self, preds,
                          bpe_token_ids):

        pred_tags, bpe_tokens = [], []
        for preds_per_sent, bpe_token_ids_per_sent in zip(preds, bpe_token_ids):

            pred_tags_per_sent, bpe_tokens_per_sent = [], []
            for pred, bpe_token_id in zip(preds_per_sent, bpe_token_ids_per_sent):
                pred_tag = self.index_to_tag[pred]
                bpe_token = self.tokenizer.convert_ids_to_tokens(int(bpe_token_id))

                if bpe_token != self.PAD:
                    pred_tags_per_sent.append(pred_tag)
                    bpe_tokens_per_sent.append(bpe_token)

            pred_tags.append(pred_tags_per_sent)
            bpe_tokens.append(bpe_tokens_per_sent)

        return pred_tags, bpe_tokens

    @staticmethod
    def transform_bpe_to_bio(pred_tags: 'list of list of str',
                             bpe_tokens: 'list of list of str',
                             bpe_token_start_idxs: 'list of list of float'):
        """transform bpe format (subword) to bio format (word)
        Args:
            pred_tags:
            bpe_tokens:
            bpe_token_start_idxs:

        Returns:
            bio_pred_tags:
            bio_tokens:
        """

        bpe_token_start_positions = []
        for bpe_tokens_per_sent, bpe_token_start_idxs_per_sent in zip(bpe_tokens, bpe_token_start_idxs):

            bpe_token_start_positions_per_sent = []
            max_length = len(bpe_tokens_per_sent)
            for position, bpe_token_start_idx in enumerate(bpe_token_start_idxs_per_sent):
                # Position cannot over the length of padded bpe_tokens of corresponding sentence,
                # So get position under the length of padded bpe_tokens of corresponding sentence
                if bpe_token_start_idx == 1.0 and position < max_length:
                    bpe_token_start_positions_per_sent.append(position)

            bpe_token_start_positions.append(bpe_token_start_positions_per_sent)

        bio_pred_tags, bio_tokens = [], []
        for i in range(len(pred_tags)):
            pred_tags_per_sent = pred_tags[i]
            bpe_tokens_per_sent = bpe_tokens[i]
            bpe_token_start_positions_per_sent = bpe_token_start_positions[i]

            bio_pred_tags_per_sent, bio_tokens_per_sent = [], []
            for j in range(len(pred_tags_per_sent)):

                if j < (len(bpe_token_start_positions_per_sent) - 1):
                    s_idx = bpe_token_start_positions_per_sent[j]
                    e_idx = bpe_token_start_positions_per_sent[j+1]

                    bio_tag = pred_tags_per_sent[s_idx]
                    bio_token = ''.join(bpe_tokens_per_sent[s_idx:e_idx])
                    bio_token = bio_token.replace("#", "")

                    bio_pred_tags_per_sent.append(bio_tag)
                    bio_tokens_per_sent.append(bio_token)

                else:
                    s_idx = bpe_token_start_positions_per_sent[j]

                    bio_tag = pred_tags_per_sent[s_idx]
                    bio_token = ''.join(bpe_tokens_per_sent[s_idx:])
                    bio_token = bio_token.replace("#", "")

                    bio_pred_tags_per_sent.append(bio_tag)
                    bio_tokens_per_sent.append(bio_token)

                    break

            bio_pred_tags.append(bio_pred_tags_per_sent)
            bio_tokens.append(bio_tokens_per_sent)

        return bio_pred_tags, bio_tokens

    @staticmethod
    def transform_bio_to_ngram(bio_pred_tags,
                               bio_tokens):
        """
        convert bio format (BIO scheme) to n-gram format

        Args:
            bio_pred_tags:
            bio_tokens:

        Returns:
            ngram_pred_tags:
            ngram_tokens:
        """

        # Get tokens that are tagged as "B" of "I", not as "O"
        bi_pred_tags, bi_tokens = [], []
        for bio_pred_tags_per_sent, bio_tokens_per_sent in zip(bio_pred_tags, bio_tokens):

            bi_pred_tags_per_sent, bi_tokens_per_sent = [], []
            for bio_pred_tag, bio_token in zip(bio_pred_tags_per_sent, bio_tokens_per_sent):
                if bio_pred_tag != "O" and bio_pred_tag != "PAD":
                    bi_pred_tags_per_sent.append(bio_pred_tag)
                    bi_tokens_per_sent.append(bio_token)

            bi_pred_tags.append(bi_pred_tags_per_sent)
            bi_tokens.append(bi_tokens_per_sent)

        # Get the position of token tagged as "B-"
        b_tag_start_positions = []
        for bi_pred_tags_per_sent in bi_pred_tags:

            b_tag_start_positions_per_sent = []
            for position, bi_pred_per_token in enumerate(bi_pred_tags_per_sent):
                if bi_pred_per_token.startswith('B'):
                    b_tag_start_positions_per_sent.append(position)

            b_tag_start_positions.append(b_tag_start_positions_per_sent)

        # combine bio tokens that are sequentially tagged as B-tag I-tags into a unified ngram
        ngram_pred_tags, ngram_tokens = [], []
        for i in range(len(bi_pred_tags)):
            bi_pred_tags_per_sent = bi_pred_tags[i]
            bi_tokens_per_sent = bi_tokens[i]
            b_tag_start_positions_per_sent = b_tag_start_positions[i]

            ngram_preds_per_sent, ngram_tokens_per_sent = [], []
            for j in range(len(bi_pred_tags_per_sent)):

                if j < (len(b_tag_start_positions_per_sent) - 1):
                    try:
                        s_idx = b_tag_start_positions_per_sent[j]
                        e_idx = b_tag_start_positions_per_sent[j+1]

                        ngram_tag = bi_pred_tags_per_sent[s_idx][2:]
                        ngram_token = '_'.join(bi_tokens_per_sent[s_idx:e_idx])
                    except:
                        ngram_tag = "O"
                        ngram_token = "<UnKnown>"

                    ngram_preds_per_sent.append(ngram_tag)
                    ngram_tokens_per_sent.append(ngram_token)

                elif j == (len(b_tag_start_positions_per_sent) - 1):
                    try:
                        s_idx = b_tag_start_positions_per_sent[j]

                        ngram_tag = bi_pred_tags_per_sent[s_idx][2:]
                        ngram_token = '_'.join(bi_tokens_per_sent[s_idx:])
                    except:
                        ngram_tag = "O"
                        ngram_token = "<UnKnown>"

                    ngram_preds_per_sent.append(ngram_tag)
                    ngram_tokens_per_sent.append(ngram_token)
                    break

                else:
                    break

            ngram_pred_tags.append(ngram_preds_per_sent)
            ngram_tokens.append(ngram_tokens_per_sent)

        return ngram_pred_tags, ngram_tokens

    def preprocess_sentence(self, bio_tokens):

        preprocessed_sents = []
        for bio_tokens_per_sent in bio_tokens:

            preprocessed_sent = []
            for bio_token in bio_tokens_per_sent:
                if bio_token == self.CLS: continue
                if bio_token == self.SEP: continue
                if bio_token == self.PAD: continue

                if bio_token.strip() != '':
                    preprocessed_sent.append(bio_token.strip())

            preprocessed_sents.append(" ".join(preprocessed_sent))

        return preprocessed_sents

    @staticmethod
    def get_span_info(ngram_pred_tags, ngram_tokens, preprocessed_sents):
        """
        get span info. of ngram token
        both ngram token and tag are the inference of DNN model, so there might be some error during transformation
        so, we need to update previously generated ngram_tokens and ngram_pred_tags
        """

        new_ngram_pred_tags = []
        new_ngram_tokens = []
        ngram_spans = []
        for i in range(len(ngram_tokens)):
            ngram_tokens_per_sent = ngram_tokens[i]
            ngram_pred_tags_per_sent = ngram_pred_tags[i]
            current_sent = preprocessed_sents[i]

            # how many times a specific ngram is mentioned in current sentence
            ngram_counter = {}
            # the next order of a specific ngram
            ngram_next_order = {}
            for j in range(len(ngram_tokens_per_sent)):
                ngram_token = ngram_tokens_per_sent[j]

                if ngram_token not in ngram_counter.keys():
                    ngram_counter[ngram_token] = 1
                    ngram_next_order[ngram_token] = 0
                else:
                    ngram_counter[ngram_token] += 1


            new_ngram_pred_tags_per_sent = []
            new_ngram_tokens_per_sent = []
            ngram_spans_per_sent = []
            for j in range(len(ngram_tokens_per_sent)):
                ngram_token = ngram_tokens_per_sent[j]
                ngram_pred_tag = ngram_pred_tags_per_sent[j]

                # it means ngram is mentioned only one time in current sentence
                if ngram_counter[ngram_token] == 1:
                    p = re.compile(ngram_token.replace("_", " "))
                    obj = p.search(current_sent)

                    if obj is not None:
                        obj_span = str(obj.start()) + "_" + str(obj.end())

                        new_ngram_pred_tags_per_sent.append(ngram_pred_tag)
                        new_ngram_tokens_per_sent.append(ngram_token)
                        ngram_spans_per_sent.append(obj_span)

                # it means ngram is mentioned more than one time in current sentence
                elif ngram_counter[ngram_token] > 1:
                    p = re.compile(ngram_token.replace("_", " "))
                    obj_iter = p.finditer(current_sent)

                    for i, next_obj in enumerate(obj_iter):
                        if i == ngram_next_order[ngram_token]:
                            obj_span = str(next_obj.start()) + "_" + str(next_obj.end())

                            new_ngram_pred_tags_per_sent.append(ngram_pred_tag)
                            new_ngram_tokens_per_sent.append(ngram_token)
                            ngram_spans_per_sent.append(obj_span)

                            ngram_next_order[ngram_token] += 1
                else:
                    raise ValueError("Unknown ngram found")

            new_ngram_pred_tags.append(new_ngram_pred_tags_per_sent)
            new_ngram_tokens.append(new_ngram_tokens_per_sent)
            ngram_spans.append(ngram_spans_per_sent)

        return new_ngram_pred_tags, new_ngram_tokens, ngram_spans