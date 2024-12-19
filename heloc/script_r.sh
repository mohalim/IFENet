#!/bin/bash
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 64 1.0 1 flatten > output_1_0.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 64 1.5 1 flatten > output_1_5.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 64 2.0 1 flatten > output_2_0.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 64 2.5 1 flatten > output_2_5.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 64 3.0 1 flatten > output_3_0.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 64 3.5 1 flatten > output_3_5.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 64 4.0 1 flatten > output_4_0.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 64 4.5 1 flatten > output_4_5.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 64 5.0 1 flatten > output_5_0.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 64 5.5 1 flatten > output_5_5.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 64 6.0 1 flatten > output_6_0.log 2>&1
python3 ifenet_heloc_optuna.py -m train -d 512 8359 0 -t 0.01 0 500 500 -cl 1 -cu 128 -f 64 6.5 1 flatten > output_6_5.log 2>&1
