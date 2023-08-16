import logging

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL


def create_model(args):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_type = args.model_type.lower()

    if args.model_type=='tresnet_m':
        model = TResnetM(model_params)
    elif args.model_type=='tresnet_l':
        model = TResnetL(model_params)
    elif args.model_type=='tresnet_xl':
        model = TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(args.model_type))
        exit(-1)

    return model
