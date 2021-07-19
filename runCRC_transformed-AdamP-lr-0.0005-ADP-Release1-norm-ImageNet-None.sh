source /ssd2/HistoKT/source/env/bin/activate
python src/adas/train.py --config /ssd2/HistoKT/source/NewPostTrainingConfigs/CRC_transformed/AdamP/None-config.yaml --output ADP-Release1_norm_ImageNet_color_aug_None_ADPL3/CRC_transformed/AdamP/output/deep_tuning/ --checkpoint ADP-Release1_norm_ImageNet_color_aug_None_ADPL3/CRC_transformed/AdamP/checkpoint/deep_tuning/lr-0.0005 --data /ssd3/users/mhosseini/datasets/ --pretrained_model /ssd2/HistoKT/source/ADPL3_weights/ImageNet.pth.tar --freeze_encoder False --save-freq 200 --norm_vals ImageNet --gpu 2

