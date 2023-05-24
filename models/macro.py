
import torch
import torch.nn as nn
import numpy as np
from argparse import Namespace
from models.resnet import resnet9
from models.vgg import vgg11
from models.sparse_resnet import sparse_resnet9
from utils import setup_logger

logger = setup_logger(__name__)

def check_list_mode_head():
    return ['fine', 'coarse', 'both']

def check_list_mode_out():
    return ['pred', 'features']

def check_modes(mode_heads, mode_out):
    if mode_heads and mode_heads not in check_list_mode_head():
        raise ValueError('Mode_head is invalid')
    if mode_out and mode_out not in check_list_mode_out():
        raise ValueError('Mode_out is invalid')

def add_head(num_features, num_classes, dropout):
    mod =nn.Sequential(
    nn.Dropout(dropout),
    nn.Linear(num_features, num_classes))
    return mod

class BilateralNet(nn.Module):
    """
    A Bilateral network with two hemispheres and two heads
    The hemispheres types are configurable.
    The heads are for fine and coarse labels respectively.
    """

    def __init__(self,
                 mode_out: str,
                 farch: str, carch: str,
                 cmodel_path = None, fmodel_path = None,
                 cfreeze_params: bool = True, ffreeze_params: bool = True,
                 fine_k: float = None, fine_per_k: float = None,
                 coarse_k: float = None, coarse_per_k: float = None,
                 dropout: float = 0.0):
        """ Initialize BilateralNet

        Args:
            carch (str): architecture for coarse labels
            farch (str): architecture for fine labels
            mode_out (str): where to get the outputs from
                                both = classifications from both heads
                                feature = features from both hemispheres (input to heads)
        """
        super(BilateralNet, self).__init__()
        
        logger.debug(f"------- Initialize BilaterallNet with farch: {farch}, carch: {carch}, mode_out: {mode_out}, dropout: {dropout}")        
        check_modes(None, mode_out)

        self.mode_out = mode_out
        
        # create the hemispheres
        self.fine_hemi = globals()[farch](Namespace(**{"k": fine_k, "k_percent": fine_per_k,}))
        self.coarse_hemi = globals()[farch](Namespace(**{"k": coarse_k, "k_percent": coarse_per_k,}))            

        # load the saved trained parameters, and freeze from further training
        if fmodel_path is not None and fmodel_path != '':
            str = "------- Load fine hemisphere"
            load_hemi_model(self.fine_hemi, fmodel_path)
            if ffreeze_params:
                freeze_params(self.fine_hemi)
                str += "      ---> and freeze"
            logger.debug(str)
        if cmodel_path is not None and cmodel_path != '':
            str = "------- Load coarse hemisphere"
            load_hemi_model(self.coarse_hemi, cmodel_path)
            if cfreeze_params:
                freeze_params(self.coarse_hemi)
                str += "     ---> and freeze"
            logger.debug(str)

        # add heads
        num_features = self.fine_hemi.num_features + self.coarse_hemi.num_features
        logger.debug(f"-- num_features: {num_features}")
        self.fine_head = add_head(num_features, 100, dropout)
        self.coarse_head = add_head(num_features, 20, dropout)

    def forward(self, x):
        fembed = self.fine_hemi(x)
        cembed = self.coarse_hemi(x)
        embed = torch.cat([fembed, cembed], axis=1)
        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out


class UnilateralNet(nn.Module):
    """
    A Unilateral network with one hemisphere and one or two heads.
    The hemisphere and head type(s) are configurable.
    The heads can be for fine, coarse or both labels respectively.
    """

    def __init__(self,
                 mode_heads: str,
                 mode_out: str,
                 arch: str,
                 model_path: str,
                 freeze_params: bool,
                 k: float, per_k: float,
                 dropout: float,
                 for_ensemble: bool = False):
        """ Initialize UnilateralNet

        Args:
            arch (str): architecture for the hemisphere
            mode_heads (str): which heads to create
                                both = create fine and coarse heads
                                fine, coarse = just fine or coarse
            mode_out (str): where to get the outputs from: pred (head output) or feature (hemisphere(s) output)
        """
        super(UnilateralNet, self).__init__()
        
        logger.debug(f"------- Initialize UnilateralNet with mode_heads: {mode_heads}, mode_out: {mode_out}, arch: {arch}, model_path: {model_path}, k: {k}, per_k: {per_k}, freeze_params: {freeze_params}, dropout: {dropout}")

        if for_ensemble and mode_heads != 'both':
            raise ValueError('For ensemble, mode_heads must be "both"')
        check_modes(mode_heads, mode_out)

        self.mode_heads = mode_heads
        self.mode_out = mode_out

        # create the hemispheres
        self.hemisphere = globals()[arch](Namespace(**{"k": k, "k_percent": per_k,}))
        
        # add heads
        num_features = self.hemisphere.num_features
        logger.debug(f"-- num_features: {num_features}")

        if self.mode_heads == 'fine' or self.mode_heads == 'both':        
            self.fine_head = add_head(num_features, 100, dropout)
        if self.mode_heads == 'coarse' or self.mode_heads == 'both':                    
            self.coarse_head = add_head(num_features, 20, dropout)

    def forward(self, x):
        embed = self.hemisphere(x)

        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            if self.mode_heads == 'fine':
                fh = self.fine_head(embed)
                return fh
            if self.mode_heads == 'coarse':
                ch = self.coarse_head(embed)
                return ch
            if self.mode_heads == 'both':
                fh = self.fine_head(embed)
                ch = self.coarse_head(embed)
                return fh, ch        


class EnsembleNet(nn.Module):
    """
    An Ensemble network with multiple UnilateralNet models.
    Assumes each one was created with two heads.
    Averages the classification outputs from all of them.
    Designed for EVALUATION ONLY of a set of trained models.
    """

    def __init__(self,
                 arch: str,
                 model_path_list: list,
                 freeze_params: bool,
                 k: float, per_k: float,
                 dropout: float):
        """ Initialize EnsembleNet

        Args:
            arch (str): architecture for the hemisphere
            mode (str): which heads to create and where to get output
                                both = create fine and coarse heads, get classification output
                                fine, coarse = just fine or coarse, get classification output
                                features = don't create heads, get output features as output
        """
        super(EnsembleNet, self).__init__()
        
        logger.debug(f"------- Initialize EnsembleNet with arch: {arch}, model_path_list: {model_path_list}, k: {k}, per_k: {per_k}, freeze_params: {freeze_params}, dropout: {dropout}")

        def create_load_model(model_path):
            model = UnilateralNet('both', 'pred', arch, model_path, freeze_params, k, per_k, dropout, for_ensemble=True)
            load_uni_model(model, model_path)
            return model

        self.model_list = nn.ModuleList([create_load_model(model_path) for model_path in model_path_list])

    def forward(self, x):
        outputs = []
        for model in self.model_list:
            outputs.append(model(x))

        avg_output_f = torch.mean(torch.stack([output[0] for output in outputs]), dim=0)
        avg_output_c = torch.mean(torch.stack([output[1] for output in outputs]), dim=0)

        return avg_output_f, avg_output_c

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False

def bilateral(args):
    """ Return a single hemisphere or bilateral (two hemispheres) network with two heads, based on the args
        See BilateralNet class for more details
    """
    return BilateralNet(args.mode_out,
                        args.farch, args.carch, 
                        args.fmodel_path, args.cmodel_path,
                        args.ffreeze_params, args.cfreeze_params,
                        args.fine_k, args.fine_per_k,
                        args.coarse_k ,args.coarse_per_k,
                        args.dropout)

def unilateral(args):
    """ Return a unilateral network (one hemisphere) with one head, based on the args
        See UnilateralNet class for more details
    """
    return UnilateralNet(args.mode_heads,
                         args.mode_out,
                         args.farch,
                         args.fmodel_path, 
                         args.ffreeze_params,
                         args.fine_k, args.fine_per_k,
                         args.dropout)

def ensemble(args):
    return EnsembleNet(args.farch,
                       args.fmodel_path, 
                       args.ffreeze_params,
                       args.fine_k, args.fine_per_k,
                       args.dropout)

def load_uni_model(model, ckpt_path):
    """ 
        Load hemisphere and heads
        The hemisphere was trained with UnilateralNet, and it's being loaded again directly 
        We need to take out the namespace variables that don't apply to the model directly
    """
    sdict = torch.load(ckpt_path)['state_dict']
    model_dict = {k.replace('model.', ''):v for k,v in sdict.items()}
    model.load_state_dict(model_dict)
    return model

def load_hemi_model(model, ckpt_path):
    """ 
        Load hemisphere (not heads) 
        The hemisphere was trained with UnilateralNet, and it's being prepared for BilateralNet
        So we need to strip out the heads (with strict=False)
        We need to take out the namespace variables that don't apply to the model directly
    """
    sdict = torch.load(ckpt_path)['state_dict']
    model_dict = {k.replace('model.', '').replace('hemisphere.', ''):v for k,v in sdict.items()}
    model.load_state_dict(model_dict, strict=False)
    return model

############## THESE ONES USED BY GRADCAM AND TEST ##############

def load_model(model, ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    # TODO this version is used by gradcam, so will need to update with new names

    sdict = torch.load(ckpt_path)['state_dict']

    model_dict = {k.replace('model.', '').replace('encoder.', ''):v for k,v in sdict.items()}
    model.load_state_dict(model_dict)
    return model

def load_bicam_model(model, ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    # TODO I belive this is used to load the bilateral model. Both hemispheres and heads.
    sdict = torch.load(ckpt_path)['state_dict']
    model_dict = {k.replace('model_', '').replace('encoder.', ''):v for k,v in sdict.items() if not 'combiner' in k and not 'fc' in k}
    fc_dict = {k.replace('combiner.', '').replace('broad.', 'ccombiner.').replace('narrow.', 'fcombiner.'):v for k,v in sdict.items() if 'combiner' in k}
    model_dict.update(fc_dict)
    model.load_state_dict(model_dict)
    return model

def load_feat_model(model, ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    # TODO not sure what encoder is for ... fix when need to use this
    sdict = torch.load(ckpt_path)['state_dict']
    model_dict = {k.replace('model.', '').replace('encoder.', ''):v for k,v in sdict.items() if 'fc' not in k}
    model.load_state_dict(model_dict)
    return model
