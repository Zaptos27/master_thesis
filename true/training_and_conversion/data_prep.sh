#!/bin/bash

export EOS_MGM_URL=root://eosuser.cern.ch
source /eos/user/m/mimodekj/env/venv/bin/activate

ls /eos/user/m/mimodekj

if [ $2 -eq 0 ]
then
	python3 data_prep.py --window_size 3
elif [ $2 -eq 1 ]
then
	python3 data_prep.py --window_size 4
elif [ $2 -eq 2 ]
then
	python3 data_prep.py --window_size 3 --backwards
else
	python3 data_prep.py --window_size 4 --backwards
fi
