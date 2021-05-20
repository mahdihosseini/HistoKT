import sys

mod_name = vars(sys.modules[__name__])['__name__']

if 'adas.' in mod_name:
    from .ADPDataset import ADPDataset
    from .MHIST import MHIST
    from .ImageNet import ImageNet
    from .TinyImageNet import TinyImageNet
    from .BACH import BACH
    from .Osteosarcoma import OSDataset
else:
    from datasets.ADPDataset import ADPDataset
    from datasets.MHIST import MHIST
    from datasets.ImageNet import ImageNet
    from datasets.TinyImageNet import TinyImageNet
    from datasets.BACH import BACH
    from datasets.Osteosarcoma import OSDataset
