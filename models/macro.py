
import torch
import torch.nn as nn
from argparse import Namespace

def check_list():
    return ['fine', 'coarse', 'both', 'feature']

class BilateralNet(nn.Module):
    """
    A Bilateral network with two hemispheres and two heads.
    The hemispheres types are configurable.
    The heads are for fine and coarse labels respectively.
    """
    def __init__(self,
                    carch: str,
                    farch: str, 
                    cmodel_path = None,
                    fmodel_path = None,
                    cfreeze_params: bool = False,
                    ffreeze_params: bool = False,
                    bilat_mode: str = 'both'):
        """ Initialize BicamNet

        Args:
            carch (str): architecture for coarse labels
            farch (str): architecture for fine labels
            bicam_mode (str): where to get the outputs from
                                both = classifications from both heads
                                feature = features from both hemispheres (input to heads)
        """
        super(BilateralNet, self).__init__()
        
        self.bilat_mode = bilat_mode
        
        # create the hemispheres
        self.broad = globals()[carch](
                                Namespace(**
                                    {
                                        "k": 0.9,
                                        "k_percent": 0.9,
                                    }))
        self.narrow = globals()[farch](
                                Namespace(**
                                    {
                                        "k": 0.9,
                                        "k_percent": 0.9,
                                    }))
        
        # load the saved trained parameters, and freeze from further training
        if cmodel_path is not None:
            self.broad.load_state_dict(load_model(cmodel_path))
            if cfreeze_params:
                freeze_params(self.broad)
        if fmodel_path is not None:
            self.narrow.load_state_dict(load_model(fmodel_path))
            if ffreeze_params:
                freeze_params(self.narrow)
        
        # add heads
        out_dim = self.broad.res2[-1][0].out_channels + self.narrow.res2[-1][0].out_channels
        self.fcombiner = nn.Sequential(
                            nn.Dropout(0.6),
                            nn.Linear(out_dim, 100)
                            )
        self.ccombiner = nn.Sequential(
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
        if self.bilat_mode == 'both':
            foutput = self.fcombiner(embed)
            coutput = self.ccombiner(embed)
            return foutput, coutput
        elif self.bilat_mode == 'feature':
            return embed

class UnilateralNet(nn.Module):
    """
    A Unilateral network with one hemisphere and one or two heads.
    The hemisphere and head type(s) are configurable.
    The heads can be for fine, coarse or both labels respectively.
    """
    def __init__(self,
                    arch: str,
                    model_path = None,
                    freeze_params: bool = False,
                    unilat_mode: str = 'both'):
        """ Initialize UnilateralNet

        Args:
            arch (str): architecture for the hemisphere
            unilat_mode (str): which heads to create and where to get output
                                both = create fine and coarse heads, get classification output
                                fine, coarse = just fine or coarse, get classification output
                                features = don't create heads, get output features as output
        """
        super(BilateralNet, self).__init__()
        
        self.unilat_mode = unilat_mode
        
        # create the hemispheres
        self.hemisphere = globals()[arch](
                                Namespace(**
                                    {
                                        "k": 0.9,
                                        "k_percent": 0.9,
                                    }))

        # load the saved trained parameters, and freeze from further training
        if model_path is not None:
            self.hemisphere.load_state_dict(load_model(model_path))
            if freeze_params:
                freeze_params(self.hemisphere)
        
        # add heads
        out_dim = self.hemisphere.res2[-1][0].out_channels

        if self.unilat_mode not in check_list():
            raise Exception('Mode of unilateral network does not match')
        
        # TODO now that this is in macro, it differs from what we did previously
        # when it was in resnet as well, and we were training a single hemisphere
        # there was no Dropout
        # so we should parameterise this and put in config, then we can replicate
        if self.unilat_mode == 'fine' or self.mode == 'both':        
            self.fine_head = nn.Sequential(
                                nn.Dropout(0.6),
                                nn.Linear(out_dim, 100)
                                )
        if self.unilat_mode == 'coarse' or self.mode == 'both':                    
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

        if self.unilat_mode == 'fine':
            fh = self.fine_head(embed)
            return fh
        if self.unilat_mode == 'coarse':
            ch = self.coarse_head(embed)
            return ch
        if self.unilat_mode == 'both':
            fh = self.fine_head(embed)
            ch = self.coarse_head(embed)
            return fh, ch
        elif self.unilat_mode == 'feature':
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
    return BilateralNet(args.carch,
                    args.farch,  
                    args.cmodel_path, args.fmodel_path, 
                    args.cfreeze_params, args.ffreeze_params,
                    args.bilat_mode)

def unilateral(args):
    """[summary]
    """
    return UnilateralNet(args.carch,
                args.arch,  
                args.model_path, 
                args.freeze_params,
                args.unilat_mode)

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
    sdict = torch.load(ckpt_path)['state_dict']
    model_dict = {k.replace('model.', '').replace('encoder.', ''):v for k,v in sdict.items() if 'fc' not in k}
    model.load_state_dict(model_dict)
    return model

