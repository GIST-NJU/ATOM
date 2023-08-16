from .google_voc import GoogleVoc
from .google_voc2 import GoogleVoc2
from .google_coco import GoogleCoco
from .google_coco2 import GoogleCoco2
from .google_coco2_partition import GoogleCoco2Partition
from .image_net_voc import ImageNetVoc
from .image_net_voc2 import ImageNetVoc2
from .image_net_coco import ImageNetCoco
from .image_net_coco2 import ImageNetCoco2
from .image_net_coco2_partition import ImageNetCoco2Partition

from .default_voc import Voc2007Classification
from .default_voc2 import Voc2007Classification2
from .default_coco import COCO2014Classification
from .default_coco2 import COCO2014Classification2
from .default_coco2_partition import COCO2014Classification2Partition


__all__ = [
    'GoogleVoc',
    'GoogleVoc2',
    'GoogleCoco',
    'GoogleCoco2',
    'GoogleCoco2Partition',
    'ImageNetVoc',
    'ImageNetVoc2',
    'ImageNetCoco',
    'ImageNetCoco2',
    'ImageNetCoco2Partition',

    'Voc2007Classification',
    'Voc2007Classification2',
    'COCO2014Classification',
    'COCO2014Classification2',
    'COCO2014Classification2Partition',
]
