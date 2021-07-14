source /ssd2/HistoKT/source/env/bin/activate
python src/adas/train.py --config /ssd2/HistoKT/source/NewPostTrainingConfigs/OSDataset_transformed/AdamP/Color-Distortion-0.4-config.yaml --output ADP_level_1_norm_ADP-Release1/OSDataset_transformed/AdamP/output/fine_tuning/Color-Distortion/distortion-0.4 --checkpoint ADP_level_1_norm_ADP-Release1/OSDataset_transformed/AdamP/checkpoint/fine_tuning/Color-Distortion/distortion-0.4/lr-0.0001 --data /ssd2/HistoKT/datasets --pretrained_model /ssd2/HistoKT/source/new-pretraining-checkpoint/None/ADP-Release1/best_trial_2_date_2021-07-13-19-43-53.pth.tar --freeze_encoder True --save-freq 200 --color_aug Color-Distortion --norm_vals ADP-Release1 --gpu 2

python src/adas/train.py --config /ssd2/HistoKT/source/NewPostTrainingConfigs/OSDataset_transformed/AdamP/Color-Distortion-0.4-config.yaml --output ADP_level_1_norm_ADP-Release1/OSDataset_transformed/AdamP/output/deep_tuning/Color-Distortion/distortion-0.4 --checkpoint ADP_level_1_norm_ADP-Release1/OSDataset_transformed/AdamP/checkpoint/deep_tuning/Color-Distortion/distortion-0.4/lr-0.0001 --data /ssd2/HistoKT/datasets --pretrained_model /ssd2/HistoKT/source/new-pretraining-checkpoint/None/ADP-Release1/best_trial_2_date_2021-07-13-19-43-53.pth.tar --freeze_encoder False --save-freq 200 --color_aug Color-Distortion --norm_vals ADP-Release1 --gpu 2

