import torch
import os
import cv2
from tqdm import tqdm
from demo.psnr_ssim_compute import calculate_psnr_ssim, calculate_psnr_ssim_pt, calculate_psnr_ssim_auto
from demo.utils import (check_dir, AverageMeter, uint2tensor, tensor2uint8, load, _get_paths_from_images)


@torch.inference_mode()
def test_on_custom_dataset(
        lr_path: str,
        hr_path: str,
        model,
        device,
        crop_border=0,
        test_y_channel=True,
        save=False,
        save_path='',
        scale=2,
        dataset='Set5',
        save_suffix='',
        lr_suffix='',
        hr_suffix=''
):
    if save:
        save_path_ = os.path.join(save_path, f'{dataset}{os.sep}x{scale}')
        check_dir(save_path_)

    lr_imgs = _get_paths_from_images(lr_path, suffix=lr_suffix)
    hr_imgs = _get_paths_from_images(hr_path, suffix=hr_suffix)

    model.to(device).eval()
    psnr, ssim = AverageMeter(), AverageMeter()

    for i, (lr_img, hr_img) in enumerate(zip(lr_imgs, hr_imgs)):
        base, ext = os.path.splitext(os.path.basename(lr_img))
        lr = cv2.imread(lr_img)[:, :, ::-1]
        hr = cv2.imread(hr_img)[:, :, ::-1]

        lr_tensor = uint2tensor(lr).to(device)
        hr_tensor = uint2tensor(hr).to(device)
        output = model(lr_tensor)

        psnr_temp, ssim_temp = calculate_psnr_ssim_auto(output, hr_tensor, crop_border=crop_border,
                                                        test_y=test_y_channel, input_order='HWC', color_order='RGB',
                                                        mode='np')
        psnr.update(psnr_temp)
        ssim.update(ssim_temp)

        # print(f'Processing {i}: LR:{lr_img} | HR:{hr_img} | PSNR/SSIM:{psnr_temp:.4f}/{ssim_temp:.4f}')

        if save:
            output_copy = tensor2uint8(output)
            cv2.imwrite(os.path.join(save_path_, f'{base}_{save_suffix}{ext}'), output_copy[:, :, ::-1])

    avg_psnr = psnr.avg
    avg_ssim = ssim.avg
    print(f'Avg PSNR:{avg_psnr} | Avg SSIM: {avg_ssim}')

    return avg_psnr, avg_ssim


@torch.inference_mode()
def test_demo(model, device, input_path, save_path, suffix=''):
    model.to(device).eval()

    if os.path.isdir(input_path):
        lr_imgs = _get_paths_from_images(input_path)
    else:
        lr_imgs = input_path if isinstance(input_path, (tuple, list)) else [input_path]

    for im in tqdm(lr_imgs):
        base, ext = os.path.splitext(os.path.basename(im))
        lr = uint2tensor(cv2.imread(im)[:, :, ::-1]).to(device)
        output = model(lr)
        sr = tensor2uint8(output)
        cv2.imwrite(os.path.join(save_path, f'{base}{suffix}{ext}'), sr[:, :, ::-1])


if __name__ == '__main__':
    from demo.TSCN import TSCN

    # root = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
    root = r'E:\Dataset\Restoration\SR\Benchmark'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = 'TSCN'

    weight = {
        '2': r'../weights/TSCN_Bicubic_X2_38.21.pth',
        '3': r'../weights/TSCN_Bicubic_X3_34.64.pth',
        '4': r'../weights/TSCN_Bicubic_X4_32.45.pth',
    }

    datasets = ['Set5', 'Set14', 'Urban100', 'Manga109', 'BSDS100']

    for s, w in weight.items():
        for dataset in datasets:
            print(f'Processing x{s} {dataset}:', end=' ')

            model = TSCN(scale=int(s), num_stages=16).to(device)
            load(w, model, 'params', False, print_=False)

            test_on_custom_dataset(
                # Custom LR/HR images dir path
                lr_path=os.path.join(root, rf'{dataset}/LRbicx{s}'),
                hr_path=os.path.join(root, rf'{dataset}/GTmod12'),
                # Selected model
                model=model,
                device=device,
                # Test PSNR/SSIM configs
                crop_border=int(s),
                test_y_channel=True,
                # Save output or not
                save=False,
                save_path=r'../results',
                scale=int(s),
                dataset=dataset,
                save_suffix=model_name
            )

    # test_demo(
    #     model=model,
    #     device=device,
    #     input_path=r'./input',
    #     save_path=r'./output',
    #     suffix='TSCN'
    # )
