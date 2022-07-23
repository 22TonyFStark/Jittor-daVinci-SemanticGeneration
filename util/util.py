import re
import importlib
import jittor
from argparse import Namespace
import numpy as np
from PIL import Image
import os
import sys
import argparse
import scipy.io as scio
import matplotlib

def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pkl' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    jittor.save(net.state_dict(), save_path)



def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pkl' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    if not os.path.exists(save_path):
        print('not find model :' + save_path + ', do not load model!')
        return net
    weights = jittor.load(save_path)
    try:
        net.load_state_dict(weights)
    except KeyError:
        print('key error, not load!')
    except RuntimeError as err:
        print(err)
        net.load_state_dict(weights, strict=False)
        print('loaded with strict=False')
    return net


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls

def feature_normalize(feature_in):
    feature_in_norm = jittor.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature_in_norm = jittor.divide(feature_in, feature_in_norm)
    return feature_in_norm

def mean_normalize(feature, dim_mean=None):
    feature = feature - feature.mean(dim=dim_mean, keepdims=True)  # center the feature
    feature_norm = jittor.norm(feature, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature = jittor.divide(feature, feature_norm)
    return feature



def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]




def natural_sort(items):
    items.sort(key=natural_keys)



@jittor.single_process_scope()
def print_current_errors(opt, epoch, i, errors, t):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    for k, v in errors.items():
        #print(v)
        #if v != 0:
        v = v.mean().float()
        message += '%s: %.3f ' % (k, v)

    print(message)
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)


def vgg_preprocess(tensor, vgg_normal_correct=False):
    if vgg_normal_correct:
        tensor = (tensor + 1) / 2
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = jittor.concat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)

    tensor_bgr_ml = tensor_bgr - jittor.array([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(1, 3, 1, 1)
    tensor_bgr_ml.stop_grad()
    tensor_rst = tensor_bgr_ml * 255
    return tensor_rst

@jittor.single_process_scope()
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

@jittor.single_process_scope()
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def weighted_l1_loss(input, target, weights):
    out = jittor.abs(input - target)
    out = out * weights.expand_as(out)
    loss = out.mean()
    return loss

def mse_loss(input, target=0):
    return jittor.mean((input - target)**2)

