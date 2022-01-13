import re

class PipelineForBPEtoNGAM:

    def __init__(self, tokenizer,
                 index_to_tag):

        self.tokenizer = tokenizer
        self.index_to_tag = index_to_tag

        self.CLS = self.tokenizer.cls_token
        self.SEP = self.tokenizer.sep_token
        self.PAD = self.tokenizer.pad_token
        self.unk = "<UNK>"

        self.b_tag = "B"
        self.o_tag = "O"
        self.pad_tag = "PAD"

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
                    e_idx = bpe_token_start_positions_per_sent[j + 1]

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

    def transform_bio_to_ngram(self, bio_pred_tags,
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
                if bio_pred_tag != self.o_tag and bio_pred_tag != self.pad_tag:
                    bi_pred_tags_per_sent.append(bio_pred_tag)
                    bi_tokens_per_sent.append(bio_token)

            bi_pred_tags.append(bi_pred_tags_per_sent)
            bi_tokens.append(bi_tokens_per_sent)

        # Get the position of token tagged as "B-"
        b_tag_start_positions = []
        for bi_pred_tags_per_sent in bi_pred_tags:

            b_tag_start_positions_per_sent = []
            for position, bi_pred_per_token in enumerate(bi_pred_tags_per_sent):
                if bi_pred_per_token.startswith(self.b_tag):
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
                        e_idx = b_tag_start_positions_per_sent[j + 1]

                        ngram_tag = bi_pred_tags_per_sent[s_idx][2:]
                        ngram_token = '_'.join(bi_tokens_per_sent[s_idx:e_idx])
                    except:
                        ngram_tag = self.o_tag
                        ngram_token = self.unk

                    ngram_preds_per_sent.append(ngram_tag)
                    ngram_tokens_per_sent.append(ngram_token)

                elif j == (len(b_tag_start_positions_per_sent) - 1):
                    try:
                        s_idx = b_tag_start_positions_per_sent[j]

                        ngram_tag = bi_pred_tags_per_sent[s_idx][2:]
                        ngram_token = '_'.join(bi_tokens_per_sent[s_idx:])
                    except:
                        ngram_tag = self.o_tag
                        ngram_token = self.unk

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
        both ngram token and tag are the inference of BERT model, so there might be some error during transformation
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