source /ssd2/HistoKT/source/env/bin/activate

python src/adas/train.py --config /ssd2/HistoKT/source/NewPretrainingConfigs/ADP-Release1-None-configAdas.yaml --output new-ADPL3-pretraining-output/None/ADP-Release1 --checkpoint new-ADPL3-pretraining-checkpoint/None/ADP-Release1 --data /ssd3/users/mhosseini/datasets/ --freeze_encoder False --save-freq 200 --norm_vals ADP-Release1 --gpu 2

