# BERT-tutorial
## Installation
`pip install pytorch-pretrained-bert==0.6.1`

## Training
```
python run_classifier.py --task_name PG --do_train --do_eval --do_lower_case --data_dir data/PG --bert_model /home/sharing/pretrained_embedding/bert/chinese_L-12_H-768_A-12 --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir /tmp/PG/
```

## Evaluation
```
cp /home/sharing/pretrained_embedding/bert/chinese_L-12_H-768_A-12/vocab.txt /tmp/PG/
python run_classifier.py --task_name PG --do_eval --do_lower_case --data_dir data/PG --bert_model /tmp/PG/ --output_dir /tmp/$(date '+%F_%H:%M:%S')
```

