#!/bin/bash
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 8 5.0 1 flatten > output_att_8.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 16 5.0 1 flatten > output_att_16.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 32 5.0 1 flatten > output_att_32.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 64 5.0 1 flatten > output_att_64.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 84 5.0 1 flatten > output_att_84.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 96 5.0 1 flatten > output_att_96.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 114 5.0 1 flatten > output_att_114.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 128 5.0 1 flatten > output_att_128.log 2>&1