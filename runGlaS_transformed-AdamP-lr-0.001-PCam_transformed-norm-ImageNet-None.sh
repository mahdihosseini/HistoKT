source /ssd2/HistoKT/source/env/bin/activate
python src/adas/train.py --config /ssd2/HistoKT/source/NewPostTrainingConfigs/GlaS_transformed/AdamP/None-config.yaml --output PCam_transformed_norm_ImageNet_color_aug_None_ImageNet/GlaS_transformed/AdamP/output/deep_tuning/ --checkpoint PCam_transformed_norm_ImageNet_color_aug_None_ImageNet/GlaS_transformed/AdamP/checkpoint/deep_tuning/lr-0.001 --data /ssd1/users/mhosseini/datasets/ --pretrained_model /ssd2/HistoKT/source/BestImageNet_Weights/PCam_transformed.pth.tar --freeze_encoder False --save-freq 200 --norm_vals ImageNet --gpu 1

