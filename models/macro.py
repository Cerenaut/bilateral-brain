
import torch
import torch.nn as nn
from argparse import Namespace
from models.resnet_v0 import resnet9

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
                 coarse_k: float = None, coarse_per_k: float = None):
        """ Initialize BicamNet

        Args:
            carch (str): architecture for coarse labels
            farch (str): architecture for fine labels
            bicam_mode (str): where to get the outputs from
                                both = classifications from both heads
                                feature = features from both hemispheres (input to heads)
        """
        super(BilateralNet, self).__init__()
        
        self.mode = mode
        
        # create the hemispheres
        self.fine_hemi = globals()[farch](
                                    Namespace(**
                                        {
                                            "k": fine_k,
                                            "k_percent": fine_per_k,
                                        }))

        self.coarse_hemi = globals()[carch](
                                    Namespace(**
                                        {
                                            "k": coarse_k,
                                            "k_percent": coarse_per_k,
                                        }))
            
        # load the saved trained parameters, and freeze from further training
        if fmodel_path is not None:
            self.narrow.load_state_dict(load_model(fmodel_path))
            if ffreeze_params:
                freeze_params(self.narrow)
        if cmodel_path is not None:
            self.broad.load_state_dict(load_model(cmodel_path))
            if cfreeze_params:
                freeze_params(self.broad)

        # add heads
        out_dim = self.broad.res2[-1][0].out_channels + self.narrow.res2[-1][0].out_channels
        self.fine_hemi = nn.Sequential(
                            nn.Dropout(0.6),
                            nn.Linear(out_dim, 100)
                            )
        self.coarse_hemi = nn.Sequential(
                            nn.Dropout(0.6),
                            nn.Linear(out_dim, 20)
                            )

    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [description]
        """
        fembed = self.narrow(x)
        cembed = self.broad(x)
        embed = torch.cat([fembed, cembed], axis=1)
        if self.mode == 'both':
            f_out = self.fine_hemi(embed)
            c_out = self.coarse_hemi(embed)
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
                 k: float, k_percent: float):
        """ Initialize UnilateralNet

        Args:
            arch (str): architecture for the hemisphere
            mode (str): which heads to create and where to get output
                                both = create fine and coarse heads, get classification output
                                fine, coarse = just fine or coarse, get classification output
                                features = don't create heads, get output features as output
        """
        super(UnilateralNet, self).__init__()
        
        self.mode = mode
        
        # create the hemispheres
        self.hemisphere = globals()[arch](
                                Namespace(**
                                    {
                                        "k": k,
                                        "k_percent": k_percent,
                                    }))

        # load the saved trained parameters, and freeze from further training
        if model_path is not None:
            self.hemisphere.load_state_dict(load_model(model_path))
            if freeze_params:
                freeze_parameters(self.hemisphere)
        
        # add heads
        out_dim = self.hemisphere.res2[-1][0].out_channels

        if self.mode not in check_list():
            raise Exception('Mode of unilateral network does not match')
        
        # TODO now that this is in macro, it differs from what we did previously
        # when it was in resnet as well, and we were training a single hemisphere
        # there was no Dropout
        # so we should parameterise this and put in config, then we can replicate
        if self.mode == 'fine' or self.mode == 'both':        
            self.fine_head = nn.Sequential(
                                nn.Dropout(0.6),
                                nn.Linear(out_dim, 100)
                                )
        if self.mode == 'coarse' or self.mode == 'both':                    
            self.coarse_head = nn.Sequential(
                                nn.Dropout(0.6),
                                nn.Linear(out_dim, 20)
                                )

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
                        args.cfreeze_params, args.ffreeze_params,
                        args.narrow_k, args.narrow_per_k,
                        args.broad_k ,args.broad_per_k)


def unilateral(args):
    """[summary]
    """
    return UnilateralNet(args.mode,
                         args.arch,
                         args.model_path, 
                         args.freeze_params,
                         args.k, args.k_percent)

def load_model(model, ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    sdict = torch.load(ckpt_path)['state_dict']
    model_dict = {k.replace('model.', '').replace('encoder.', ''):v for k,v in sdict.items()}
    model.load_state_dict(model_dict)
    return model

def load_bilat_model(model, ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    #TODO have to update these names, according to new consistent naming
    sdict = torch.load(ckpt_path)['state_dict']
    model_dict = {k.replace('model_', '').replace('encoder.', ''):v for k,v in sdict.items() if not 'combiner' in k and not 'fc' in k}
    fc_dict = {k.replace('combiner.', '').replace('broad.', 'ccombiner.').replace('narrow.', 'fcombiner.'):v for k,v in sdict.items() if 'combiner' in k}
    model_dict.update(fc_dict)
    model.load_state_dict(model_dict)
    return model

def load_unicam_model(ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    sdict = torch.load(ckpt_path)
    # ['state_dict']
    model_dict = {k.replace('model.', ''):v for k,v in sdict.items() if 'conv' in k}
    return model_dict

def load_feat_model(model, ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    #TODO have to update these names, according to new consistent naming
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

