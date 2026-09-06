import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FiveTaskWrapperResNet(nn.Module):
    '''
    ResNet50 only supports single-task classification
    To enable multi-task anatomy & correction classification, we need:
    (1) ResNet50 as feature extractor backbone
    (2) Build parallel classification heads for prediction tasks
    (3) We can still unfreeze and fine-tune ResNet blocks
    '''
    def __init__(self, backbone, in_features, num_ana, num_mag, num_gain, num_centering, num_shadow):
        super().__init__()
        self.backbone = backbone
        self.head_ana = nn.Linear(in_features, num_ana)
        self.head_mag = nn.Linear(in_features, num_mag)
        self.head_gain = nn.Linear(in_features, num_gain)
        self.head_centering = nn.Linear(in_features, num_centering)
        self.head_shadow = nn.Linear(in_features, 1) # binary classification

    def forward(self, x):
        features = self.backbone(x)
        logits_ana = self.head_ana(features)
        logits_mag = self.head_mag(features)
        logits_gain = self.head_gain(features)
        logits_centering = self.head_centering(features)
        logits_shadow = self.head_shadow(features)
        return logits_ana, logits_mag, logits_gain, logits_centering, logits_shadow
    

class FourTaskWrapperResNet(nn.Module):
    '''
    ResNet50 only supports single-task classification
    To enable multi-task anatomy & correction classification, we need:
    (1) ResNet50 as feature extractor backbone
    (2) Build parallel classification heads for prediction tasks
    (3) We can still unfreeze and fine-tune ResNet blocks
    '''
    def __init__(self, backbone, in_features, num_mag, num_gain, num_centering, num_shadow):
        super().__init__()
        self.backbone = backbone
        self.head_mag = nn.Linear(in_features, num_mag)
        self.head_gain = nn.Linear(in_features, num_gain)
        self.head_centering = nn.Linear(in_features, num_centering)
        self.head_shadow = nn.Linear(in_features, 1) # binary classification

    def forward(self, x):
        features = self.backbone(x)
        logits_mag = self.head_mag(features)
        logits_gain = self.head_gain(features)
        logits_centering = self.head_centering(features)
        logits_shadow = self.head_shadow(features)
        return logits_mag, logits_gain, logits_centering, logits_shadow
    

class SonoNet(nn.Module):
    """
    PyTorch implementation of SonoNet.

    Baumgartner et al., "Real-Time Detection and Localisation of Fetal Standard
    Scan Planes in 2D Freehand Ultrasound", arXiv preprint:1612.05601 (2016)

    Based on https://github.com/baumgach/SonoNet-weights which
    provides a theano+lasagne implementation.
    
    Real-Time Detection of Freehand Fetal Ultrasound Standard Scan Planes

    PyTorch implementation of the original models.py file, plus functions to
    load and convert the lasagne weights to a PyTorch state_dict.

    Args:
        config (str): Selects the architecture.
            Options are 'SN16', 'SN32' or 'SN64'
        num_labels (int, optional): Length of output vector after adaption.
            Default is 14. Ignored if features_only=True
        weights (bool, 0 or string): Select weight initialization.
            True: Load weights from default *.pth weight file.
            False: No weights are initialized.
            0: Standard random weight initialization.
            str: Pass your own weight file.
            Default is True.
        features_only (bool, optional): If True, only feature layers are
            initialized and the forward method returns the features.
            Default is False.

    Attributes:
        feature_channels (int): Number of feature channels.
        features (torch.nn.Sequential): Feature extraction CNN
        adaption (torch.nn.Sequential): Adaption layers for classification

    Examples::
    net = sononet.SonoNet('SN64').eval().cuda()
    outputs = net(x)

    encoder =
                sononet.SonoNet('SN64', features_only=True).eval().cuda()
     features = encoder(x)

    Note:
        Inputs into the forward methods must be preprocessed as shown test.py
    """

    feature_cfg_dict = {
        'SN16': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M',
                 128, 128, 128, 'M', 128, 128, 128],
        'SN32': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M',
                 256, 256, 256, 'M', 256, 256, 256],
        'SN64': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                 512, 512, 512, 'M', 512, 512, 512]
    }

    def __init__(self, config, num_labels=14, weights=True,
                 features_only=False, in_channels=1):
        super().__init__()
        self.config = config
        self.feature_cfg = self.feature_cfg_dict[config]
        self.feature_channels = self.feature_cfg[-1]
        self.weights = weights
        self.features_only = features_only
        self.features = self._make_feature_layers(self.feature_cfg, in_channels)
        if not features_only:
            self.adaption_channels = self.feature_channels // 2
            self.num_labels = num_labels
            self.adaption = self._make_adaption_layer(
                self.feature_channels, self.adaption_channels, self.num_labels)
        self.set_weights(weights)

    def forward(self, x):
        x = self.features(x)
        if not self.features_only:
            x = self.adaption(x)
            x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
            #x = F.softmax(x, dim=1)
        return x

    @staticmethod
    def _make_adaption_layer(feature_channels, adaption_channels, num_labels):
        return nn.Sequential(
            nn.Conv2d(feature_channels,
                      adaption_channels, 1, bias=False),
            nn.BatchNorm2d(adaption_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(adaption_channels, num_labels, 1, bias=False),
            nn.BatchNorm2d(num_labels),
        )

    def set_weights(self, weights):
        if weights is not None:
            if weights:
                if not isinstance(weights, str):
                    weights = os.path.join(
                        os.path.dirname(__file__),
                        'SonoNet{}.pth'.format(self.config[2:]))
                self.load_weights(weights)
            else:
                self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    @staticmethod
    def _conv_layer(in_channels, out_channels):
        layer = [nn.Conv2d(in_channels, out_channels,
                           kernel_size=3, padding=1, bias=False),
                 nn.BatchNorm2d(out_channels, eps=1e-4),
                 nn.ReLU(inplace=True)]
        return nn.Sequential(*layer)

    @classmethod
    def _make_feature_layers(cls, feature_cfg, in_channels):
        layers = []
        conv_layers = []
        for v in feature_cfg:
            if v == 'M':
                conv_layers.append(nn.MaxPool2d(2))
                layers.append(nn.Sequential(*conv_layers))
                conv_layers = []
            else:
                conv_layers.append(cls._conv_layer(in_channels, v))
                in_channels = v
        layers.append(nn.Sequential(*conv_layers))
        return nn.Sequential(*layers)

    @staticmethod
    def process_lasagne_weights(weights):
        order = [0, 2, 1, 3, 4]
        weights = [weights[5 * (idx // 5) + order[idx % 5]]
                   for idx in range(len(weights))]
        weights[4::5] = [np.power(w, -2) - 1e-4 for w in weights[4::5]]
        return weights

    @classmethod
    def load_lasagne_weights(cls, filename, state):
        with np.load(filename) as f:
            weight_data = [f['arr_%d' % i] for i in range(len(f.files))]
        weight_data = cls.process_lasagne_weights(weight_data)
        offset = 0
        for idx, layer in enumerate(state):
            if 'num_batches_tracked' in layer:
                # TODO: Initialize to different value?
                offset -= 1
                continue
            # assert tuple(state[layer].shape) == weight_data[idx].shape
            if not tuple(state[layer].shape) == weight_data[idx + offset].shape:
                pass
            state[layer] = torch.from_numpy(weight_data[idx + offset].copy())

    @staticmethod
    def save_state(state, filename):
        if (not os.path.isfile(filename) or
                input('Overwrite state file?\nHit [y] to continue: ') == 'y'):
            torch.save(state, filename)

    def load_weights(self, weights):
        _, extension = os.path.splitext(weights)
        if extension == '.npz':
            state = self.state_dict()
            state = self.load_lasagne_weights(weights, state)
            self.save_state(
                state, os.path.join(os.path.dirname(__file__), 'SonoNet{}.pth'.format(self.config[2:])))
        elif extension == '.pth':
            state = torch.load(weights)
        else:
            raise ValueError('Unknown weight file extension {}'
                             .format(extension))
        # Check if input channels match
        # for key in self.state_dict():
        #     size = self.state_dict()[key].size()
        #     if state[key].size() != size:
        #         expand = state[key].expand(size)
        #         state[key] = expand * state[key].norm() / expand.norm()
        # self.load_state_dict(state, strict=True)


class FiveTaskWrapperSonoNet(nn.Module):
    def __init__(self, backbone, num_ana, num_mag, num_gain, num_centering, num_shadow):
        super().__init__()
        self.backbone = backbone.features  # shared feature extractor

        # optional shared pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # feature dimension from SN64 is 512
        feat_dim = 512

        self.head_ana = nn.Linear(feat_dim, num_ana)
        self.head_mag = nn.Linear(feat_dim, num_mag)
        self.head_gain = nn.Linear(feat_dim, num_gain)
        self.head_centering = nn.Linear(feat_dim, num_centering)
        self.head_shadow = nn.Linear(feat_dim, num_shadow)   # binary classification

    def forward(self, x):
        x = self.backbone(x)          # [B, C, H, W]
        x = self.pool(x)              # [B, C, 1, 1]
        x = torch.flatten(x, 1)       # [B, C]

        logits_ana = self.head_ana(x)
        logits_mag = self.head_mag(x)
        logits_gain = self.head_gain(x)
        logits_centering = self.head_centering(x)
        logits_shadow = self.head_shadow(x) # binary logit

        return logits_ana, logits_mag, logits_gain, logits_centering, logits_shadow

class FourTaskWrapperSonoNet(nn.Module):
    def __init__(self, backbone, num_mag, num_gain, num_centering, num_shadow):
        super().__init__()
        self.backbone = backbone.features  # shared feature extractor

        # optional shared pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # feature dimension from SN64 is 512
        feat_dim = 512

        self.head_mag = nn.Linear(feat_dim, num_mag)
        self.head_gain = nn.Linear(feat_dim, num_gain)
        self.head_centering = nn.Linear(feat_dim, num_centering)
        self.head_shadow = nn.Linear(feat_dim, num_shadow)   # binary classification

    def forward(self, x):
        x = self.backbone(x)          # [B, C, H, W]
        x = self.pool(x)              # [B, C, 1, 1]
        x = torch.flatten(x, 1)       # [B, C]

        logits_mag = self.head_mag(x)
        logits_gain = self.head_gain(x)
        logits_centering = self.head_centering(x)
        logits_shadow = self.head_shadow(x) # binary logit

        return logits_mag, logits_gain, logits_centering, logits_shadow    
    

#################################################
### Vision Transformer helper functions start ###
#################################################
# All vision transformers need these following classes to work. 
# For the vision transformer to be ViT-B/16, you need to set hyperparameters to the ViT-B/16 defaults:
# IMAGE_SIZE = 224 # usually 224 but can be another number divisible by PATCH__SIZE
# PATCH_SIZE = 16
# EMBED_DIM = 768
# NUM_HEADS = 12
# DEPTH = 12
# MLP_DIM = 3072
# Then, you must initialise the model using (example for OneTaskVisionTransformer): 
# model = VisionTransformer(
#     img_size=IMAGE_SIZE,
#     patch_size=PATCH_SIZE,
#     in_channels=CHANNELS,
#     num_classes=NUM_CLASSES,
#     embed_dim=EMBED_DIM,
#     num_heads=NUM_HEADS,
#     depth=DEPTH,
#     mlp_dim=MLP_DIM,
#     drop_rate=DROP_RATE
# ).to(device)
class PatchEmbedding(nn.Module):
  def __init__(self, img_size, patch_size, in_channels, embed_dim):
    super().__init__()
    self.patch_size = patch_size
    self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
    num_patches = (img_size // patch_size) ** 2
    self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

  def forward(self, x:torch.Tensor):
    B = x.size(0)
    x = self.proj(x) # (B, E, Height/Patch_Size, Width/Patch_Size)
    x = x.flatten(2).transpose(1, 2) # (B, N, E)
    cls_token = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = x + self.pos_embed
    return x

class MLP(nn.Module):
  def __init__(self, in_features, hidden_features, drop_rate):
    super().__init__()
    self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
    self.fc2 = nn.Linear(in_features=hidden_features, out_features=in_features)
    self.dropout = nn.Dropout(drop_rate)

  def forward(self, x):
    x = self.fc1(x)
    x = F.gelu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.dropout(x)
    return x
  
class TransformerEncoderLayer(nn.Module):
  def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate):
    super().__init__()
    self.norm1 = nn.LayerNorm(embed_dim)
    self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=drop_rate, batch_first=True)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_dim, drop_rate=drop_rate)

  def forward(self, x):
    x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
    x = x + self.mlp(self.norm2(x))
    return x
###############################################
### Vision Transformer helper functions end ###
###############################################

class OneTaskVisionTransformer(nn.Module):
  def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, depth, mlp_dim, drop_rate):
    super().__init__()
    self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
    self.encoder = nn.Sequential(*[ # python will run everything in this bracket sequentially. "*" is needed to prevent error.
        TransformerEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, drop_rate=drop_rate)
        for _ in range(depth) # if depth=10, python will loop through TransformerEncoderLayer 10 times to create 10 Transformer Encoders.
    ])
    self.norm = nn.LayerNorm(embed_dim)
    self.head = nn.Linear(embed_dim, num_classes)

  def forward(self, x):
    x = self.patch_embed(x)
    x = self.encoder(x)
    x = self.norm(x)
    cls_token = x[:,0]
    x = self.head(cls_token)
    return x

class FiveTaskVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        num_ana,
        num_mag,
        num_gain,
        num_centering,
        num_shadow,
        embed_dim,
        num_heads,
        depth,
        mlp_dim,
        drop_rate
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.encoder = nn.Sequential(*[
            TransformerEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, drop_rate=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # 4 multiclass classification heads
        self.head_ana = nn.Linear(embed_dim, num_ana)
        self.head_mag = nn.Linear(embed_dim, num_mag)
        self.head_gain = nn.Linear(embed_dim, num_gain)
        self.head_centering = nn.Linear(embed_dim, num_centering)
        # 1 binary classification head: output 1 logit
        self.head_shadow = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]  # (B, embed_dim)
        logits_ana = self.head_ana(cls_token)
        logits_mag = self.head_mag(cls_token)
        logits_gain = self.head_gain(cls_token)
        logits_centering = self.head_centering(cls_token)
        logits_shadow = self.head_shadow(cls_token)  # (B, 1)
        return logits_ana, logits_mag, logits_gain, logits_centering, logits_shadow

class FiveTaskWrapperViT(nn.Module):
    def __init__(self, backbone, num_ana, num_mag, num_gain, num_centering, num_shadow):
        super().__init__()
        self.backbone = backbone
        self.head_ana = nn.Linear(768, num_ana)
        self.head_mag = nn.Linear(768, num_mag)
        self.head_gain = nn.Linear(768, num_gain)
        self.head_centering = nn.Linear(768, num_centering)
        self.head_shadow = nn.Linear(768, 1)

    def forward(self, x):
        # backbone returns CLS embedding
        feat = self.backbone(x)
        logits_ana = self.head_ana(feat)
        logits_mag = self.head_mag(feat)
        logits_gain = self.head_gain(feat)
        logits_centering = self.head_centering(feat)
        logits_shadow = self.head_shadow(feat)
        return (logits_ana, logits_mag, logits_gain, logits_centering, logits_shadow)
    

class FourTaskVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        num_mag,
        num_gain,
        num_centering,
        num_shadow,
        embed_dim,
        num_heads,
        depth,
        mlp_dim,
        drop_rate
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.encoder = nn.Sequential(*[
            TransformerEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, drop_rate=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # 4 multiclass classification heads
        self.head_mag = nn.Linear(embed_dim, num_mag)
        self.head_gain = nn.Linear(embed_dim, num_gain)
        self.head_centering = nn.Linear(embed_dim, num_centering)
        # 1 binary classification head: output 1 logit
        self.head_shadow = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]  # (B, embed_dim)
        logits_mag = self.head_mag(cls_token)
        logits_gain = self.head_gain(cls_token)
        logits_centering = self.head_centering(cls_token)
        logits_shadow = self.head_shadow(cls_token)  # (B, 1)
        return logits_mag, logits_gain, logits_centering, logits_shadow

class FourTaskWrapperViT(nn.Module):
    def __init__(self, backbone, num_mag, num_gain, num_centering, num_shadow):
        super().__init__()
        self.backbone = backbone
        self.head_mag = nn.Linear(768, num_mag)
        self.head_gain = nn.Linear(768, num_gain)
        self.head_centering = nn.Linear(768, num_centering)
        self.head_shadow = nn.Linear(768, 1)

    def forward(self, x):
        # backbone returns CLS embedding
        feat = self.backbone(x)
        logits_mag = self.head_mag(feat)
        logits_gain = self.head_gain(feat)
        logits_centering = self.head_centering(feat)
        logits_shadow = self.head_shadow(feat)
        return (logits_mag, logits_gain, logits_centering, logits_shadow)
    