#!/bin/bash
ltrain='/content/drive/My Drive/bosch_dataset/train.yaml'
ltest='/content/drive/My Drive/bosch_dataset/test.yaml'
use_riib=True
batch_size=5
data_path='/content/drive/My Drive/bosch_dataset/data_reduced'
start_eval=1
output_path='/content/drive/My Drive/models'
device='cpu'
num_workers=1
epochs=100

python trainer.py -ltrain "$ltrain" \
-ltest "$ltest" \
--num_workers $num_workers \
--batch_size $batch_size \
--use_riib $use_riib \
--data_path "$data_path" \
--start_eval $start_eval \
--output_path "$output_path" \
--device $device \
--epochs $epochs
