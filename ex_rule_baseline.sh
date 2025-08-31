#!/bin/bash

python bola.py
python bb.py
python quetra.py

python run_rmpc_c_version.py
python run_bs_mpc.py bs
python run_bs_mpc.py mfd

./dp_my