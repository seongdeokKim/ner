# ner
Below line is the code for executing paper which is currently under-review.

For finetuning
    python ner_finetune.py --model_type bert_bilstm --model_fn ./simple_ntc/models/bert_bilstm.pth --train_fn ./data/ner_dataset.txt --gpu_id 0 --batch_size 16 --n_epochs 6 --lr 5e-5 --max_length 64
For prediction
    cat ner_sample.txt | python ner_predict.py --model_fn ./simple_ntc/models/bert_bilstm.pth --gpu_id 0 > ner_result.txt
