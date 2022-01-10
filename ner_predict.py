import sys
import argparse

import torch
import numpy as np

from ner_finetune import get_model
from utils.data_loader import TokenizerForInference, PipelineForBPEtoNGAM



def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=1024)

    config = p.parse_args()

    return config


def read_text():
    '''
    Read text from standard input for inference.
    '''
    lines = []
    for line in sys.stdin:
       if line.strip() != '':
           lines += [line.strip()]

    return lines


def main(config):

    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:{}'.format(config.gpu_id)
    )

    train_config = saved_data['config']
    best_model = saved_data['model']
    loaded_tokenizer = saved_data['tokenizer']
    index_to_tag = saved_data['index_to_tag']

    lines = read_text()

    with torch.no_grad():
        # Declare model and load fine-tuned weights.
        #tokenizer = AutoTokenizer.from_pretrained(train_config.pretrained_model_name)
        tokenizer = TokenizerForInference(loaded_tokenizer,
                                          train_config.max_length,
                                          index_to_tag)

        pipeline = PipelineForBPEtoNGAM(loaded_tokenizer,
                                        index_to_tag)

        model = get_model(train_config, index_to_tag)
        model.load_state_dict(best_model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        bpe_token_ids, preds = [], []
        bpe_token_start_idxs = []
        
        # Turn-on evaluation mode.
        model.eval()
        for idx in range(0, len(lines), config.batch_size):

            mini_batch = tokenizer.tokenize(lines[idx:idx + config.batch_size])
            input_ids = mini_batch['input_ids']
            input_ids = input_ids.to(device)
            attention_mask = mini_batch['attention_mask']
            attention_mask = attention_mask.to(device)
            _bpe_token_start_idxs = mini_batch['bpe_token_start_idxs']

            if 'crf' not in train_config.model_type:
                outputs = model(input_ids,
                                attention_mask=attention_mask)

                logits = outputs[0]

                logits = logits.detach().cpu().numpy()
                input_ids = input_ids.detach().cpu().numpy()

                for preds_per_sent in np.argmax(logits, axis=2):
                    preds += [preds_per_sent]
                for input_ids_per_sent in input_ids:
                    bpe_token_ids += [input_ids_per_sent]
                for _bpe_token_start_idxs_per_sent in _bpe_token_start_idxs:
                    bpe_token_start_idxs += [_bpe_token_start_idxs_per_sent]

            elif 'crf' in train_config.model_type:
                outputs = model(input_ids,
                                attention_mask=attention_mask)

                sequence_of_tags = outputs[0]

                sequence_of_tags = sequence_of_tags.to('cpu').numpy()
                input_ids = input_ids.detach().cpu().numpy()

                for preds_per_sent in sequence_of_tags:
                    preds += [preds_per_sent]
                for input_ids_per_sent in input_ids:
                    bpe_token_ids += [input_ids_per_sent]
                for _bpe_token_start_idxs_per_sent in _bpe_token_start_idxs:
                    bpe_token_start_idxs += [_bpe_token_start_idxs_per_sent]

        result = pipeline.run(preds,
                              bpe_token_ids,
                              bpe_token_start_idxs)


        for i in range(len(lines)):
            sys.stdout.write('{}\t{}\t{}\t{}\t{}\n'.format(
                "|".join(result["ngram_tokens"][i]),
                "|".join(result["ngram_pred_tags"][i]),
                "|".join(result["ngram_spans"][i]),
                result["preprocessed_sents"][i],
                lines[i]
            ))



if __name__ == '__main__':

    config = define_argparser()
    main(config)
