import sys

mod_name = vars(sys.modules[__name__])['__name__']

if 'adas.' in mod_name:
    from .ADP_dataset import ADP_dataset
    from .MHIST import MHIST
    from .ImageNet import ImageNet
    from .TinyImageNet import TinyImageNet
else:
    from datasets.ADP_dataset import ADP_dataset
    from datasets.MHIST import MHIST
    from datasets.ImageNet import ImageNet
    from datasets.TinyImageNet import TinyImageNet