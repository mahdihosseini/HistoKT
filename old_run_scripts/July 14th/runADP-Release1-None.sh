source /ssd2/HistoKT/source/env/bin/activate
python src/adas/train.py --config /ssd2/HistoKT/source/NewPretrainingConfigs/ADP-Release1-None-configAdas.yaml --output new-pretraining-output/None/ADP-Release1 --checkpoint new-pretraining-checkpoint/None/ADP-Release1 --data /ssd2/HistoKT/datasets --save-freq 200