# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.



from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset

class SceneDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=29)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        
        subfolder = 'val_' if opt.phase == 'test' else 'all_'
        cache = False if opt.phase == 'test' else True
        if opt.phase != "test":
            image_paths = sorted(make_dataset(root + subfolder + "img", recursive=True, read_cache=cache, write_cache=False))
            label_paths = sorted(make_dataset(root + subfolder + "label", recursive=True, read_cache=cache, write_cache=False))
        else:
            image_paths = sorted(
                #make_dataset("/home/qingzhongfei/A_scene/630CODE_v1/data/B", recursive=True, read_cache=cache, write_cache=False)
                make_dataset(opt.trainimg_root, recursive=True, read_cache=cache, write_cache=False)
                #make_dataset("/home/user/duzongwei/Projects/JTGAN/SPADE/datasets/train/train_img", recursive=True, read_cache=cache, write_cache=False)
            )
            label_paths = sorted(make_dataset(opt.input_path, recursive=True, read_cache=cache, write_cache=False))
            #label_paths = sorted(make_dataset("/home/user/duzongwei/Projects/JTGAN/SPADE/datasets/val_A_labels_cleaned/", recursive=True, read_cache=cache, write_cache=False))

        return label_paths, image_paths

    def get_ref(self, opt):
        extra = 'test' if opt.phase == 'test' else 'train'
        
        #with open('./data/scene_ref_{}_bestiou_v1.txt'.format(extra)) as fd:
        #with open('./data/scene_ref_{}_iou_v1.txt'.format(extra)) as fd:
        #with open('./data/scene_ref_{}_iou_v1_debug.txt'.format(extra)) as fd:
        if opt.phase == 'test':
            fname = './data/scene_ref_testB_bestiou_v1.txt'
        else:
            fname = './data/scene_ref_{}_iou_v1.txt'.format(extra)
        with open(fname) as fd:
            lines = fd.readlines()
        ref_dict = {}
        for i in range(len(lines)):
            items = lines[i].strip().split(',')
            items.pop(-1)
            key = items[0]
            if opt.phase == 'test':
                val = items[1:]
            else:
                val = [items[1], items[-1]]
            ref_dict[key] = val
        train_test_folder = ('training', 'validation')
        return ref_dict, train_test_folder

    def imgpath_to_labelpath(self, path):
        path = path.replace("img","label")
        return path


