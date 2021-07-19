source /ssd2/HistoKT/source/env/bin/activate
python src/adas/train.py --config /ssd2/HistoKT/source/NewPostTrainingConfigs/BCSS_transformed/AdamP/None-config.yaml --output ADP-Release1_norm_ADP-Release1_color_aug_None_ADPL3/BCSS_transformed/AdamP/output/deep_tuning/ --checkpoint ADP-Release1_norm_ADP-Release1_color_aug_None_ADPL3/BCSS_transformed/AdamP/checkpoint/deep_tuning/lr-0.001 --data /ssd1/users/mhosseini/datasets/ --pretrained_model /ssd2/HistoKT/source/ADPL3_weights/ADP-Release1.pth.tar --freeze_encoder False --save-freq 200 --norm_vals ADP-Release1 --gpu 0

