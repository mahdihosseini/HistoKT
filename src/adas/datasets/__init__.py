import sys

mod_name = vars(sys.modules[__name__])['__name__']

if 'adas.' in mod_name:
    from .ADPDataset import ADPDataset
    from .AIDPATH import AIDPATH
    from .AJLymph import AJLymph
    from .BACH import BACH
    from .CRC import CRC
    from .GlaS import GlaS
    from .MHIST import MHIST
    from .Osteosarcoma import OSDataset
    from .PCam import PCam
    from .ImageNet import ImageNet
    from .TinyImageNet import TinyImageNet
    from .TransformedDataset import TransformedDataset
    from .BCSSDataset import BCSSDataset
else:
    from datasets.ADPDataset import ADPDataset
    from datasets.AIDPATH import AIDPATH
    from datasets.AJLymph import AJLymph
    from datasets.BACH import BACH
    from datasets.CRC import CRC
    from datasets.GlaS import GlaS
    from datasets.MHIST import MHIST
    from datasets.Osteosarcoma import OSDataset
    from datasets.PCam import PCam
    from datasets.ImageNet import ImageNet
    from datasets.TinyImageNet import TinyImageNet
    from datasets.TransformedDataset import TransformedDataset
    from datasets.BCSSDataset import BCSSDataset