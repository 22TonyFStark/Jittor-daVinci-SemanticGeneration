# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import jittor
import jittor.nn as nn
from util.util import feature_normalize
import numpy as np
from jittor import transform as transforms


postpa = transforms.Compose([
    transforms.Lambda(lambda x: x * (1. / 255)),
    transforms.ImageNormalize(
        mean=[-0.40760392, -0.45795686, -0.48501961],  #add imagenet mean
        std=[1, 1, 1]),
    transforms.Lambda(lambda x: x[jittor.array([2, 1, 0]).stop_grad()]),  #turn to RGB
])
postpb = transforms.Compose([transforms.ToPILImage()])


def post_processing(tensor):
    t = postpa(tensor)  # denormalize the image since the optimized tensor is the normalized one
    t[t > 1] = 1
    t[t < 0] = 0
    img = postpb(t)
    img = np.array(img)
    return img


class ContextualLoss(nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self):
        super(ContextualLoss, self).__init__()
        return None

    def execute(self, X_features, Y_features, h=0.1, feature_centering=True):
        '''
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        '''
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        feature_size = X_features.shape[2]

        # center the feature vector

        # to normalized feature vectors
        if feature_centering:
            X_features = X_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
            Y_features = Y_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        X_features = feature_normalize(X_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size^2
        Y_features = feature_normalize(Y_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size^2

        # conine distance = 1 - similarity
        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        d = 1 - jittor.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2

        # normalized distance: dij_bar
        d_norm = d / (jittor.min(d, dim=-1, keepdims=True) + 1e-5)  # batch_size * feature_size^2 * feature_size^2

        # pairwise affinity
        w = jittor.exp((1 - d_norm) / h)
        A_ij = w / jittor.sum(w, dim=-1, keepdims=True)

        # contextual loss per sample
        CX = jittor.mean(jittor.max(A_ij, dim=1), dim=-1)
        loss = -jittor.log(CX)

        return loss


class ContextualLoss_forward(nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self, opt):
        super(ContextualLoss_forward, self).__init__()
        self.opt = opt
        return None

    def execute(self, X_features, Y_features, h=0.1, feature_centering=True):
        '''
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        '''
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        feature_size = X_features.shape[2]

        # to normalized feature vectors
        if feature_centering:
            if self.opt.PONO:
                X_features = X_features - Y_features.mean(dim=1).unsqueeze(dim=1)
                Y_features = Y_features - Y_features.mean(dim=1).unsqueeze(dim=1)
            else:
                X_features = X_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
                Y_features = Y_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        X_features = feature_normalize(X_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size * feature_size
        Y_features = feature_normalize(Y_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size * feature_size


        # conine distance = 1 - similarity
        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        d = 1 - jittor.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2

        # normalized distance: dij_bar
        # d_norm = d
        d_norm = d / (jittor.min(d, dim=-1, keepdims=True) + 1e-3)  # batch_size * feature_size^2 * feature_size^2

        # pairwise affinity
        w = jittor.exp((1 - d_norm) / h)
        A_ij = w / jittor.sum(w, dim=-1, keepdims=True)

        # contextual loss per sample
        CX = jittor.mean(jittor.max(A_ij, dim=-1), dim=1)
        loss = -jittor.log(CX)

        return loss


class ContextualLoss_complex(nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self):
        super(ContextualLoss_complex, self).__init__()
        return None

    def execute(self, X_features, Y_features, h=0.1, patch_size=1, direction='forward'):
        '''
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        '''
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        feature_size = X_features.shape[2]

        # to normalized feature vectors
        X_features = X_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        Y_features = Y_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        X_features = feature_normalize(X_features)  # batch_size * feature_depth * feature_size^2
        Y_features = feature_normalize(Y_features)  # batch_size * feature_depth * feature_size^2

        # to normalized feature vectors
        X_features = nn.unfold(
            X_features, kernel_size=(patch_size, patch_size), stride=(1, 1), padding=(patch_size // 2,
                                                                                      patch_size // 2))  # batch_size * feature_depth_new * feature_size^2
        Y_features = nn.unfold(
            Y_features, kernel_size=(patch_size, patch_size), stride=(1, 1), padding=(patch_size // 2,
                                                                                      patch_size // 2))  # batch_size * feature_depth_new * feature_size^2

        # conine distance = 1 - similarity
        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        d = 1 - jittor.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2

        # normalized distance: dij_bar
        d_norm = d / (jittor.min(d, dim=-1, keepdims=True) + 1e-5)  # batch_size * feature_size^2 * feature_size^2

        # pairwise affinity
        w = jittor.exp((1 - d_norm) / h)
        A_ij = w / jittor.sum(w, dim=-1, keepdims=True)

        # contextual loss per sample
        if direction == 'execute':
            CX = jittor.mean(jittor.max(A_ij, dim=-1), dim=1)
        else:
            CX = jittor.mean(jittor.max(A_ij, dim=1), dim=-1)

        loss = -jittor.log(CX)
        return loss


if __name__ == "__main__":
    contextual_loss = ContextualLoss()
    batch_size = 32
    feature_depth = 8
    feature_size = 16
    X_features = jittor.zeros((batch_size, feature_depth, feature_size, feature_size))
    Y_features = jittor.zeros((batch_size, feature_depth, feature_size, feature_size))

    cx_loss = contextual_loss(X_features, Y_features, 1)
    print(cx_loss)