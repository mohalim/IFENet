#!/bin/bash
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 16 -f 128 5.0 1 flatten > output_hidden_16.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 24 -f 128 5.0 1 flatten > output_hidden_24.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 32 -f 128 5.0 1 flatten > output_hidden_32.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 64 -f 128 5.0 1 flatten > output_hidden_64.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 84 -f 128 5.0 1 flatten > output_hidden_84.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 96 -f 128 5.0 1 flatten > output_hidden_96.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 114 -f 128 5.0 1 flatten > output_hidden_114.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 128 5.0 1 flatten > output_hidden_128.log 2>&1
