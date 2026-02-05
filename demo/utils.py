import torch
import numpy as np
import os
from natsort import os_sorted
import time



def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def uint2tensor(img):
    img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.).float().unsqueeze(0)
    return img


def tensor2uint8(img):
    img = img.detach().cpu().numpy().astype(np.float32).squeeze(0).transpose(1, 2, 0)
    img = np.uint8((img.clip(0., 1.) * 255.).round())
    return img


def load(path, model, key='state_dict', delete_module=False, delete_str='module.', print_=True):
    from collections import OrderedDict

    def delete_state_module(weights, delete_str='module.'):
        weights_dict = {}
        for k, v in weights.items():
            # new_k = k.replace('module.', '') if 'module' in k else k
            new_k = k.replace(delete_str, '') if delete_str in k else k
            weights_dict[new_k] = v
        return weights_dict

    ckpt = torch.load(path, map_location=lambda storage, loc: storage)  # ['state_dict']
    ckpt = ckpt.get(key, ckpt)  # ckpt if key == '' else ckpt[key]  #

    if delete_module:
        ckpt = delete_state_module(ckpt, delete_str)

    overlap = OrderedDict()
    for key, value in ckpt.items():
        if key in model.state_dict().keys() and ckpt[key].shape == model.state_dict()[key].shape:
            overlap[key] = value
        else:
            try:
                print(f'Failed load: ckpt: {key}-{ckpt[key].shape} | model: {key}-{model.state_dict()[key].shape}')
            except Exception as e:
                print(f'Failed load: ckpt: {key}-{ckpt[key].shape}, maybe not in model_state!')

    if print_:
        print(f'{(len(overlap) * 1.0 / len(ckpt) * 100):.4f}% weights is loaded!', end='\t')
        print(f'{(len(overlap) * 1.0 / len(model.state_dict()) * 100):.4f}% params is inited!')

    try:
        model.load_state_dict(overlap, strict=True)
        if print_: print(f'Loading weights from {path}.')
    except RuntimeError as e:
        print(f'RuntimeError: {e}')


def _get_paths_from_images(path, suffix=''):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname) and suffix in fname:
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return os_sorted(images)


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

if __name__ == '__main__':
    from collections import OrderedDict
    ckpt = torch.load(r'../weights/TSCN_RealSR_X4.pth', map_location=lambda storage, loc: storage)['state_dict']

    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        new_state_dict[k.replace('tail_.', 'tail.')] = v

    torch.save({'params': new_state_dict}, r'../weights/TSCN_RealSR_X4.pth')


    # for k, v in ckpt.items():
    #     print(k)