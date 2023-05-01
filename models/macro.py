
import torch
import torch.nn as nn
from argparse import Namespace
from models.resnet import resnet9
from utils import setup_logger

def check_list():
    return ['fine', 'coarse', 'both', 'feature']

class BilateralNet(nn.Module):
    """
    A Bilateral network with two hemispheres and two heads.
    The hemispheres types are configurable.
    The heads are for fine and coarse labels respectively.
    """

    def __init__(self, mode: str,
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
            mode (str): where to get the outputs from
                                both = classifications from both heads
                                feature = features from both hemispheres (input to heads)
        """
        super(BilateralNet, self).__init__()
        
        self.logger = setup_logger(__name__)
        self.logger.debug(f"------- Initialize BilaterallNet with farch: {farch}, carch: {carch}, mode: {mode}, dropout: {dropout}")
        
        self.mode = mode
        
        # create the hemispheres
        self.fine_hemi = globals()[farch](Namespace(**{"k": fine_k, "k_percent": fine_per_k,}))
        self.coarse_hemi = globals()[farch](Namespace(**{"k": coarse_k, "k_percent": coarse_per_k,}))            

        # load the saved trained parameters, and freeze from further training
        if fmodel_path is not None:
            str = "------- Load fine hemisphere"
            load_hemi_model(self.fine_hemi, fmodel_path)
            if ffreeze_params:
                freeze_params(self.fine_hemi)
                str += ",      ---> and freeze"
            self.logger.debug(str)
        if cmodel_path is not None:
            str = "------- Load coarse hemisphere"
            load_hemi_model(self.coarse_hemi, cmodel_path)
            if cfreeze_params:
                freeze_params(self.coarse_hemi)
                str += ",      ---> and freeze"
            self.logger.debug(str)

        # add heads
        # out_dim = self.fine_hemi.res2[-1][0].out_channels + self.coarse_hemi.res2[-1][0].out_channels
        self.fine_head = nn.Sequential(
                            nn.Dropout(dropout),
                            nn.Linear(1024, 100))
        self.coarse_head = nn.Sequential(
                            nn.Dropout(dropout),
                            nn.Linear(1024, 20))

    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [description]
        """
        fembed = self.fine_hemi(x)
        cembed = self.coarse_hemi(x)
        embed = torch.cat([fembed, cembed], axis=1)
        if self.mode == 'both':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out
        elif self.mode == 'feature':
            return embed

class UnilateralNet(nn.Module):
    """
    A Unilateral network with one hemisphere and one or two heads.
    The hemisphere and head type(s) are configurable.
    The heads can be for fine, coarse or both labels respectively.
    """

    def __init__(self,
                 mode: str,
                 arch: str,
                 model_path: str,
                 freeze_params: bool,
                 k: float, per_k: float,
                 dropout: float):
        """ Initialize UnilateralNet

        Args:
            arch (str): architecture for the hemisphere
            mode (str): which heads to create and where to get output
                                both = create fine and coarse heads, get classification output
                                fine, coarse = just fine or coarse, get classification output
                                features = don't create heads, get output features as output
        """
        super(UnilateralNet, self).__init__()
        
        self.logger = setup_logger()
        self.logger.debug(f"------- Initialize UnilateralNet with mode: {mode}, arch: {arch}, model_path: {model_path}, k: {k}, per_k: {per_k}, freeze_params: {freeze_params}, dropout: {dropout}")

        self.mode = mode
        
        # create the hemispheres
        self.hemisphere = globals()[arch](Namespace(**{"k": k, "k_percent": per_k,}))

        # load the saved trained parameters, and freeze from further training
        if model_path is not None:
            self.logger.debug(f"------- Load hemisphere from checkpoint: {model_path}")
            load_hemi_model(self.hemisphere, model_path)
            if freeze_params:
                self.logger.debug(f"------- Freeze hemisphere parameters")
                freeze_parameters(self.hemisphere)
        
        # add heads
        # out_dim = self.hemisphere.res2[-1][0].out_channels

        if self.mode not in check_list():
            raise Exception('Mode of unilateral network does not match')
        
        if self.mode == 'fine' or self.mode == 'both':        
            self.fine_head = nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Linear(512, 100))
        if self.mode == 'coarse' or self.mode == 'both':                    
            self.coarse_head = nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Linear(512, 20))

    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [description]
        """
        embed = self.hemisphere(x)

        if self.mode == 'fine':
            fh = self.fine_head(embed)
            return fh
        if self.mode == 'coarse':
            ch = self.coarse_head(embed)
            return ch
        if self.mode == 'both':
            fh = self.fine_head(embed)
            ch = self.coarse_head(embed)
            return fh, ch
        elif self.mode == 'feature':
            return embed

def freeze_params(model):
    """[summary]

    Args:
        model ([type]): [description]
    """
    for param in model.parameters():
        param.requires_grad = False
    
def bilateral(args):
    """[summary]
    """
    return BilateralNet(args.mode,
                        args.farch, args.carch, 
                        args.fmodel_path, args.cmodel_path,
                        args.ffreeze_params, args.cfreeze_params,
                        args.fine_k, args.fine_per_k,
                        args.coarse_k ,args.coarse_per_k,
                        args.dropout)

def unilateral(args):
    """[summary]
    """
    return UnilateralNet(args.mode,
                         args.arch,
                         args.model_path, 
                         args.freeze_params,
                         args.k, args.per_k,
                         args.dropout)

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

def load_hemi_model(model, ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    sdict = torch.load(ckpt_path)['state_dict']
    model_dict = {k.replace('model.', '').replace('hemisphere.', ''):v for k,v in sdict.items()}
    model.load_state_dict(model_dict, strict=False)
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

def freeze_parameters(model):
    """[summary]

    Args:
        model ([type]): [description]
    """
    for param in model.parameters():
        param.requires_grad = False

