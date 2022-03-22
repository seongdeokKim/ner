# ner
this is for paper

---- for finetuning ----
python ner_finetune.py --model_type bert_bilstm --model_fn ./simple_ntc/models/test.pth --train_fn ./data/ner_dataset.txt --gpu_id 0 --batch_size 16 --n_epochs 6 --lr 5e-5 --max_length 64

---- for prediction ----
cat ner_sample.txt | python ner_predict.py --model_fn ./simple_ntc/models/kobert.pth --gpu_id 0 > test.txt
