source /ssd2/HistoKT/source/env/bin/activate
python src/adas/train.py --config /ssd2/HistoKT/source/NewPretrainingConfigs/CRC_transformed-None-configAdas.yaml --output new-pretraining-output/None/CRC_transformed --checkpoint new-pretraining-checkpoint/None/CRC_transformed --data /ssd2/HistoKT/datasets --save-freq 200