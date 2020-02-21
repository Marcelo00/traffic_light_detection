#!/bin/bash
ltrain='/content/drive/My Drive/bosch_dataset/train.yaml'
ltest='/content/drive/My Drive/bosch_dataset/test.yaml'
use_riib=True
batch_size_train=5
batch_size_test=5
data_path='/content/drive/My Drive/bosch_dataset/balanced_data'
start_eval=10
output_path='/content/drive/My Drive/models'
device='cuda'
num_workers=1
epochs=300

python trainer.py -ltrain "$ltrain" \
-ltest "$ltest" \
--num_workers $num_workers \
--batch_size_train $batch_size_train \
--batch_size_test $batch_size_test \
--use_riib $use_riib \
--data_path "$data_path" \
--start_eval $start_eval \
--output_path "$output_path" \
--device $device \
--epochs $epochs
