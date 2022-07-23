"""
Implementation of Content loss, Style loss, LPIPS and DISTS metrics
References:
    .. [1] Gatys, Leon and Ecker, Alexander and Bethge, Matthias
    (2016). A Neural Algorithm of Artistic Style}
    Association for Research in Vision and Ophthalmology (ARVO)
    https://arxiv.org/abs/1508.06576
    .. [2] Zhang, Richard and Isola, Phillip and Efros, et al.
    (2018) The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
    2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
    https://arxiv.org/abs/1801.03924
"""
from typing import List, Union, Collection

import jittor
import jittor.nn as nn
from jittor.models import vgg16, vgg19

from .utils import _validate_input, _reduce
from .functional import similarity_map, L2Pool2d
from .reduction import legacy_get_string



class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


# Map VGG names to corresponding number in jittor layer
VGG16_LAYERS = {
    "conv1_1": '0', "relu1_1": '1',
    "conv1_2": '2', "relu1_2": '3',
    "pool1": '4',
    "conv2_1": '5', "relu2_1": '6',
    "conv2_2": '7', "relu2_2": '8',
    "pool2": '9',
    "conv3_1": '10', "relu3_1": '11',
    "conv3_2": '12', "relu3_2": '13',
    "conv3_3": '14', "relu3_3": '15',
    "pool3": '16',
    "conv4_1": '17', "relu4_1": '18',
    "conv4_2": '19', "relu4_2": '20',
    "conv4_3": '21', "relu4_3": '22',
    "pool4": '23',
    "conv5_1": '24', "relu5_1": '25',
    "conv5_2": '26', "relu5_2": '27',
    "conv5_3": '28', "relu5_3": '29',
    "pool5": '30',
}

VGG19_LAYERS = {
    "conv1_1": '0', "relu1_1": '1',
    "conv1_2": '2', "relu1_2": '3',
    "pool1": '4',
    "conv2_1": '5', "relu2_1": '6',
    "conv2_2": '7', "relu2_2": '8',
    "pool2": '9',
    "conv3_1": '10', "relu3_1": '11',
    "conv3_2": '12', "relu3_2": '13',
    "conv3_3": '14', "relu3_3": '15',
    "conv3_4": '16', "relu3_4": '17',
    "pool3": '18',
    "conv4_1": '19', "relu4_1": '20',
    "conv4_2": '21', "relu4_2": '22',
    "conv4_3": '23', "relu4_3": '24',
    "conv4_4": '25', "relu4_4": '26',
    "pool4": '27',
    "conv5_1": '28', "relu5_1": '29',
    "conv5_2": '30', "relu5_2": '31',
    "conv5_3": '32', "relu5_3": '33',
    "conv5_4": '34', "relu5_4": '35',
    "pool5": '36',
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Constant used in feature normalization to avoid zero division
EPS = 1e-10


class ContentLoss(_Loss):
    r"""Creates Content loss that can be used for image style transfer or as a measure for image to image tasks.
    Uses pretrained VGG models from jittor.models.
    Expects input to be in range [0, 1] or normalized with ImageNet statistics into range [-1, 1]

    Args:
        feature_extractor: Model to extract features or model name: ``'vgg16'`` | ``'vgg19'``.
        layers: List of strings with layer names. Default: ``'relu3_3'``
        weights: List of float weight to balance different layers
        replace_pooling: Flag to replace MaxPooling layer with AveragePooling. See references for details.
        distance: Method to compute distance between features: ``'mse'`` | ``'mae'``.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        mean: List of float values used for data standardization. Default: ImageNet mean.
            If there is no need to normalize data, use [0., 0., 0.].
        std: List of float values used for data standardization. Default: ImageNet std.
            If there is no need to normalize data, use [1., 1., 1.].
        normalize_features: If true, unit-normalize each feature in channel dimension before scaling
            and computing distance. See references for details.

    References:
        Gatys, Leon and Ecker, Alexander and Bethge, Matthias (2016).
        A Neural Algorithm of Artistic Style
        Association for Research in Vision and Ophthalmology (ARVO)
        https://arxiv.org/abs/1508.06576

        Zhang, Richard and Isola, Phillip and Efros, et al. (2018)
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
        IEEE/CVF Conference on Computer Vision and Pattern Recognition
        https://arxiv.org/abs/1801.03924
    """

    def __init__(self, feature_extractor="vgg16", layers=("relu3_3",),
                 weights=[1.], replace_pooling=False,
                 distance="mse", reduction="mean", mean=IMAGENET_MEAN,
                 std=IMAGENET_STD, normalize_features=False,
                 allow_layers_weights_mismatch=False):

        assert allow_layers_weights_mismatch or len(layers) == len(weights), \
            f'Lengths of provided layers and weighs mismatch ({len(weights)} weights and {len(layers)} layers), ' \
            f'which will cause incorrect results. Please provide weight for each layer.'

        super().__init__()

        if callable(feature_extractor):
            self.model = feature_extractor
            self.layers = layers
        else:
            if feature_extractor == "vgg16":
                self.model = vgg16(pretrained=True).features
                self.layers = [VGG16_LAYERS[l] for l in layers]
            elif feature_extractor == "vgg19":
                self.model = vgg19(pretrained=True).features
                self.layers = [VGG19_LAYERS[l] for l in layers]
            else:
                raise ValueError("Unknown feature extractor")

        if replace_pooling:
            self.model = self.replace_pooling(self.model)

        # Disable gradients
        for param in self.model.parameters():
            param.stop_grad()

        self.distance = {
            "mse": lambda x, y : (x-y).sqr(),
            "mae": lambda x, y : (x-y).abs(),
        }[distance]
        # self.distance = {
        #     "mse": nn.MSELoss,
        #     "mae": nn.L1Loss,
        # }[distance](reduction='none')

        self.weights = [jittor.Var(w).stop_grad() if not isinstance(w, jittor.Var) else w for w in weights]
        

        mean = jittor.Var(mean).stop_grad()
        std = jittor.Var(std).stop_grad()
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)

        self.normalize_features = normalize_features
        self.reduction = reduction

    def execute(self, x, y):
        r"""Computation of Content loss between feature representations of prediction :math:`x` and
        target :math:`y` tensors.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Content loss between feature representations
        """
        _validate_input([x, y], dim_range=(4, 4), data_range=(0, -1))

        # print(self.model)
        # self.model.float64()
        # print(x, y)
        x_features = self.get_features(x)
        y_features = self.get_features(y)
        # for i, j in zip(x_features, y_features):
        #     print(i.shape, j.shape)
        

        distances = self.compute_distance(x_features, y_features)
        # for i in distances:
            # print(i.shape)

        # Scale distances, then average in spatial dimensions, then stack and sum in channels dimension
        # loss = jittor.concat([(d * w.to(d)).mean(dim=[2, 3]) for d, w in zip(distances, self.weights)], dim=1).sum(dim=1)
        # print(distances, self.weights)
        # print(self.weights)
        # for d, w in zip(distances, self.weights):
            # print((d * w).shape)
        loss = jittor.concat([(d * w).mean(dims=[2, 3]) for d, w in zip(distances, self.weights)], dim=1).sum(dim=1)
        # print([(d * w).shape for d, w in zip(distances, self.weights)])
        # loss = jittor.concat([(d * w).mean() for d, w in zip(distances, self.weights)], dim=1).sum(dim=1)

        return _reduce(loss, self.reduction)

    def compute_distance(self, x_features, y_features):
        r"""Take L2 or L1 distance between feature maps depending on ``distance``.

        Args:
            x_features: Features of the input tensor.
            y_features: Features of the target tensor.

        Returns:
            Distance between feature maps
        """
        return [self.distance(x, y) for x, y in zip(x_features, y_features)]

    def get_features(self, x):
        r"""
        Args:
            x: Tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            List of features extracted from intermediate layers
        """
        # Normalize input
        # print(self.mean.dtype, x.dtype)
        x = (x - self.mean) / self.std

        features = []
        for name, module in self.model.named_modules()[1:]:
            x = module(x)
            if name in self.layers: 
                features.append(self.normalize(x) if self.normalize_features else x)

        return features

    @staticmethod
    def normalize(x):
        r"""Normalize feature maps in channel direction to unit length.

        Args:
            x: Tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Normalized input
        """
        norm_factor = jittor.sqrt(jittor.sum(x ** 2, dim=1, keepdims=True))
        return x / (norm_factor + EPS)


    
    def replace_pooling(self, module):
        r"""Turn All MaxPool layers into AveragePool
        Args:
            module: Module to change MaxPool int AveragePool
        Returns:
            Module with AveragePool instead MaxPool
        """
        new_net = jittor.nn.Sequential()
        
        for name, child in module.named_modules()[1:]:
            # print(child)
            if isinstance(child, jittor.nn.Pool):
                child = jittor.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            new_net.add_module(name, child)
        return new_net


class DISTS(ContentLoss):
    r"""Deep Image Structure and Texture Similarity metric.

    By default expects input to be in range [0, 1], which is then normalized by ImageNet statistics into range [-1, 1].
    If no normalisation is required, change `mean` and `std` values accordingly.

    Args:
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        mean: List of float values used for data standardization. Default: ImageNet mean.
            If there is no need to normalize data, use [0., 0., 0.].
        std: List of float values used for data standardization. Default: ImageNet std.
            If there is no need to normalize data, use [1., 1., 1.].

    References:
        Keyan Ding, Kede Ma, Shiqi Wang, Eero P. Simoncelli (2020).
        Image Quality Assessment: Unifying Structure and Texture Similarity.
        https://arxiv.org/abs/2004.07728
        https://github.com/dingkeyan93/DISTS
    """

    def __init__(self, reduction="mean", mean=IMAGENET_MEAN,
                 std=IMAGENET_STD):
        dists_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        channels = [3, 64, 128, 256, 512, 512]

        # 超参数
        weights = jittor.load("/home/qingzhongfei/A_scene/jittor628/models/piq/dists_weights.pkl")
        dists_weights = list(jittor.misc.split(jittor.Var(weights['alpha']).stop_grad(), channels, dim=1))
        dists_weights.extend(jittor.misc.split(jittor.Var(weights['beta']).stop_grad(), channels, dim=1))

        super().__init__("vgg16", layers=dists_layers, weights=dists_weights,
                         replace_pooling=True, reduction=reduction, mean=mean, std=std,
                         normalize_features=False, allow_layers_weights_mismatch=True)

    def execute(self, x, y):
        r"""

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Deep Image Structure and Texture Similarity loss, i.e. ``1-DISTS`` in range [0, 1].
        """
        _, _, H, W = x.shape

        if min(H, W) > 256:
            x = nn.interpolate(
                x, scale_factor=256 / min(H, W), recompute_scale_factor=False, mode='bilinear')
            y = nn.interpolate(
                y, scale_factor=256 / min(H, W), recompute_scale_factor=False, mode='bilinear')

        loss = super().execute(x, y)
        return 1 - loss

    def compute_distance(self, x_features, y_features):
        r"""Compute structure similarity between feature maps

        Args:
            x_features: Features of the input tensor.
            y_features: Features of the target tensor.

        Returns:
            Structural similarity distance between feature maps
        """
        structure_distance, texture_distance = [], []
        # Small constant for numerical stability
        EPS = 1e-6

        for x, y in zip(x_features, y_features):
            x_mean = x.mean([2, 3], keepdims=True)
            y_mean = y.mean([2, 3], keepdims=True)
            structure_distance.append(similarity_map(x_mean, y_mean, constant=EPS))

            x_var = ((x - x_mean) ** 2).mean([2, 3], keepdims=True)
            y_var = ((y - y_mean) ** 2).mean([2, 3], keepdims=True)
            xy_cov = (x * y).mean([2, 3], keepdims=True) - x_mean * y_mean
            texture_distance.append((2 * xy_cov + EPS) / (x_var + y_var + EPS))

        return structure_distance + texture_distance

    def get_features(self, x):
        r"""

        Args:
            x: Input tensor

        Returns:
            List of features extracted from input tensor
        """
        features = super().get_features(x)

        features.insert(0, x)
        return features
    
    def replace_pooling(self, module):
        r"""Turn All MaxPool layers into AveragePool
        Args:
            module: Module to change MaxPool int AveragePool
        Returns:
            Module with AveragePool instead MaxPool
        """
        new_net = jittor.nn.Sequential()
        
        for name, child in module.named_modules()[1:]:
            # print(child)
            if isinstance(child, jittor.nn.Pool):
                child = L2Pool2d(kernel_size=3, stride=2, padding=1)
            new_net.add_module(name, child)
        return new_net
    
    
    
    
if __name__ == '__main__':
    loss = ContentLoss()
    x = jittor.rand(3, 3, 256, 256)
    y = jittor.rand(3, 3, 256, 256).stop_grad()
    output = loss(x, y)
    print("contentloss", output)
    
    loss = DISTS()
    x = jittor.rand(3, 3, 256, 256)
    y = jittor.rand(3, 3, 256, 256).stop_grad()
    output = loss(x, y)
    print("dists", output)
    