# from src.adas.datasets import ADPDataset, TransformedDataset
# import os

# root_dir = "/home/zhan8425/scratch/HistoKTdata"
#
# output_dict = {}
# for dataset in ["ADP",
#                   "AIDPATH_transformed",
#                   "AJ-Lymph_transformed",
#                   "BACH_transformed",
#                   "CRC_transformed",
#                   "GlaS_transformed",
#                   "MHIST_transformed",
#                   "OSDataset_transformed",
#                   "PCam_transformed"]:
#     output_dict[dataset] = {}
#     for split in ["train", "valid", "test"]:
#         if dataset == "ADP":
#             output_dict[dataset][split] = len(ADPDataset(level="L3Only", root=root_dir, split=split))
#         else:
#             output_dict[dataset][split] = len(TransformedDataset(root=os.path.join(root_dir, dataset), split=split))
#
# print(output_dict)
# for dataset, values in output_dict.items():
#     print(f"{dataset}: train: {values['train']}, valid: {values['valid']}, test: {values['test']}")
