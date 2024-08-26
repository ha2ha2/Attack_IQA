import os
from turtle import TurtleGraphicsError

from sympy.polys.polyconfig import query

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torchvision.transforms as transforms
import scipy.stats
import numpy as np
from tqdm import tqdm
from TReS import TReS
from BaseCNN import BaseCNN
from DBCNN import DBCNN
import pandas as pd

import argparse
from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from datetime import datetime
from mapping2 import logistic_mapping
import random
from LIQE import LIQE
import pyimgsaliency as psal
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage.util import view_as_windows
import torch
# from scipy.fftpack import dct, idct
import torch_dct.dct as torch_dct





class Logger():
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()


def predict_quality_score(x, args):
    s = model(x).squeeze(1)

    return s


def random_sign(size):
    return torch.sign(-1 + 2 * torch.rand(size=size))



def get_canny(image):
    # canny map: binary array
    # image shape:[H, W, C]
    if image.shape[0] == 3:
        image = np.transpose(image*255, (1, 2, 0)).astype('uint8')
    gray = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
    low_threshold = 100
    high_threshold = 200
    canny = cv2.Canny(gray, low_threshold, high_threshold)
    return canny

def get_saliency(image):
    # binary_sal: binary map
    # image shape:[H, W, C]
    if image.shape[0] == 3:
        image = np.transpose(image*255, (1, 2, 0)).astype('uint8')
    mbd = psal.get_saliency_mbd(image).astype('uint8')
    binary_sal = psal.binarise_saliency_map(mbd, method='adaptive')
    return binary_sal

def get_zig_zag_mask(frequence_range, mask_shape):
    rows = mask_shape[0]
    cols = mask_shape[1]
    mask = torch.zeros([mask_shape[0],mask_shape[1]])

    for i in range(rows):
         for j in range(cols):
            if i + j < cols - 1:
                mask[i, j] = 1

    return mask

def get_mask(image):
    canny_mask = get_canny(image)
    sal_mask = get_saliency(image)
    mask = np.logical_or(canny_mask, sal_mask).astype(int)
    mask_3chan = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    return mask_3chan



def image_decomposition(f, lambd=0.8):
    u = denoise_tv_chambolle(f, weight=lambd)
    v = f - u
    return u, v

def compute_Cs(image, neighborhood_size=5):
    """ Calculate Cs (maximum luminance difference) in a given neighborhood. """
    # Convert image to float and normalize
    image = image.astype(float)

    # Add padding to the image to handle borders
    pad_width = neighborhood_size // 2
    padded_image = np.pad(image, pad_width=pad_width, mode='edge')

    # Create a window view
    window_shape = (neighborhood_size, neighborhood_size)
    windows = view_as_windows(padded_image, window_shape, step=1)

    # Calculate maximum and minimum values within each window
    max_vals = np.max(windows, axis=(-1, -2))
    min_vals = np.min(windows, axis=(-1, -2))

    # Calculate Cs
    Cs = max_vals - min_vals

    return Cs

def calculate_CM(u, v, beta=0.117, We=1, Wt=3):
    """ Calculate CM(x, y) based on the given images u and v. """
    # Compute Cs for u and v
    Cs_u = compute_Cs(u)
    Cs_v = compute_Cs(v)

    # Calculate EMu and TMv
    EMu = Cs_u * beta * We
    TMv = Cs_v * beta * Wt

    # Calculate CM
    CM = EMu + TMv

    return CM


def luminance_adaptation(image):
    LA = np.zeros_like(image)
    cond = image <= 127
    LA[cond] = 17 * (1 - np.sqrt(image[cond] / 127)) + 3
    LA[~cond] = 3 * (image[~cond] - 127) / 128 + 3
    return LA


def compute_jnd(image, Clc=0.3):
    if image.shape[0] == 3:
        image = np.transpose(image.cpu().numpy()*255, (1, 2, 0)).astype('uint8')
    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
    u, v = image_decomposition(image)
    CM = calculate_CM(u, v)
    LA = luminance_adaptation(image)
    JND = LA + CM - Clc * np.minimum(LA, CM)
    return JND/225.0

def get_dtex(targ_shape, kernel_size = 3):
    Inat_list = ['60', '71', ]
    Inat_name = random.choice(Inat_list)
    filename = './Inat/I'+Inat_name+'.png'
    Inat = cv2.imread(filename)
    Inat = cv2.cvtColor(Inat, cv2.COLOR_BGR2RGB)
    blurred_Inat = cv2.GaussianBlur(Inat, (kernel_size, kernel_size), 0)
    Ihfre = Inat - blurred_Inat
    Ihfre = Ihfre / 255.0
    dtex = cv2.resize(Ihfre, (targ_shape[1], targ_shape[0]))

    return dtex

def project_to_JND_1(x_1minus1, u_i, x_ori, jnd_map):
    diff = x_1minus1 + u_i - x_ori
    diff_temp = torch.clamp(diff, -jnd_map, jnd_map)
    proj_miu = u_i + (diff_temp - diff)
    return proj_miu

def project_to_JND_2(x_theta_i, x_ori, jnd_map):
    diff = x_theta_i - x_ori
    diff_temp = torch.clamp(diff, -jnd_map, jnd_map)
    proj_x_theta_i = x_theta_i + (diff_temp - diff)
    return proj_x_theta_i

def is_success(gamma, score_ori, score_pert):
    if score_ori <= 5:
        if score_pert > score_ori + gamma*(10-score_ori) :
            return True
        else:
            return False
    else:
        if score_pert < score_ori + gamma*(0-score_ori) :
            return True
        else:
            return False




def gram_schmidt_orthogonalization(u, v):
    u_dot_v = torch.mul(u, v).sum()
    proj = u_dot_v * u
    v_orthogonal = v - proj
    return v_orthogonal

def get_vi(img_ori, u_i, dct_mask):
    processed_image = get_dct(img_ori, dct_mask)
    v_i = processed_image
    v_i_orthogonal = gram_schmidt_orthogonalization(u_i, v_i)
    v_i_orthogonal = v_i_orthogonal / torch.norm(v_i_orthogonal)
    return v_i_orthogonal

def test_point(start_point, theta, u, v, di):
    temp = torch.cos(theta * np.pi / 180) * u + torch.sin(theta * np.pi / 180) * v
    new_dis = temp * di * torch.cos(theta * np.pi / 180)
    return start_point + new_dis

Q_max = 8000
def Single_Attack(img_ori, score_ori, start_point ,gamma_i, gamma_i_minus_1, gamma_i_minus_2, d_tex, combined_mask, jnd_map, T_toal, dct_mask):
    T = 0
    T_max = 200
    while(T_toal < Q_max):
        tau = np.random.uniform(-0.1, 0.1)
        u_i_hat = tau * d_tex * combined_mask
        u_i_hat = u_i_hat.transpose([2, 0, 1])
        u_i_hat = torch.from_numpy(u_i_hat.astype(np.float32)).to(device)
        u_i_hat_proj = project_to_JND_1(start_point, u_i_hat, img_ori, jnd_map)
        di = torch.norm(u_i_hat_proj)
        u_i = u_i_hat_proj / di
        x_pert = start_point + di * u_i
        x_pert = torch.clamp(x_pert, 0, 1)
        score_pert = predict_quality_score(x_pert.unsqueeze(0), args)
        logger.log('score_pert = {}'.format(score_pert))
        T += 1
        T_toal += 1
        if is_success(gamma_i, score_ori, score_pert) == False or di < 1:
            if (T > T_max):
                gamma_i = gamma_i - (gamma_i - gamma_i_minus_1) / 2
                T = 0
                continue
            else:
                continue
        else:
            break
    if is_success(gamma_i + 2*(gamma_i - gamma_i_minus_1), score_ori, score_pert):
        gamma_i = gamma_i + (gamma_i - gamma_i_minus_1)
    # v_i = get_vi(img_ori, u_i)
    xi = start_point
    min_distance = float('inf')
    best_theta = 0

    theta_max = torch.Tensor([10]).to(device)
    while (T_toal < Q_max and best_theta == 0):
        v_i = get_vi(img_ori, u_i, dct_mask)
        for sign_theta in (1, -1):
            theta_test = sign_theta * theta_max
            x_evo = test_point(start_point, theta_test, u_i, v_i, di)
            x_evo = torch.clip(x_evo, 0, 1)
            T_toal += 1
            score_pert = predict_quality_score(x_evo.unsqueeze(0), args)
            if is_success(gamma_i, score_ori, score_pert):
                best_theta = theta_test
                break
    if T_toal >= Q_max:
        return xi, gamma_i, T_toal
    lower = best_theta
    upper = lower + torch.sign(lower)*theta_max
    check_opposite = lower > 0
    while (T_toal < Q_max and abs(upper - lower) > 0.01):
        theta_mid = (lower + upper) / 2

        mid_evo = test_point(start_point, theta_mid, u_i, v_i, di)
        mid_evo = torch.clip(mid_evo, 0, 1)
        T_toal += 1
        score_pert = predict_quality_score(mid_evo.unsqueeze(0), args)
        logger.log('score_pert = {}'.format(score_pert))
        # γi-success
        if is_success(gamma_i, score_ori, score_pert):
            lower = theta_mid
            continue
        opp_theta_mid = -theta_mid
        opp_mid_evo = test_point(start_point, opp_theta_mid, u_i, v_i, di)
        opp_mid_evo = torch.clip(opp_mid_evo, 0, 1)
        T_toal += 1
        score_pert = predict_quality_score(opp_mid_evo.unsqueeze(0), args)
        logger.log('score_pert = {}'.format(score_pert))
        # γi-success
        if is_success(gamma_i, score_ori, score_pert):
            lower = -theta_mid
            upper = -upper
            continue
        lower = lower
        upper = theta_mid

    best_point = test_point(start_point, lower, u_i, v_i, di)
    best_point = torch.clip(best_point, 0, 1)
    T_toal += 1
    score_pert = predict_quality_score(best_point.unsqueeze(0), args)
    logger.log('score_pert = {}'.format(score_pert))
    check = is_success(gamma_i, score_ori, score_pert)
    xi = project_to_JND_2(best_point, img_ori, jnd_map)
    return xi, gamma_i, T_toal


def attackIQA(dataset_loader, args):
    gt = []
    x_pert_bests = []
    x_oris = []
    for i, (image, mos) in enumerate(dataset_loader):
        x = image.to(device)
        x_oris.append(x)
        gt.append(mos.cpu().numpy())
        x_pert_bests.append(x)

    logger.log('start optimize......')
    for n in range(args.n_sample // args.bs):
        # a batch of images: x_oris[n]
        x_ori, f_x = x_oris[n].clone(), fx[n]
        # images in one batch
        # sample new delta for each img at random position
        for i_img in tqdm(range(x_ori.shape[0])):
            # i_img: one image
            gamma_i_minus_1 = 0.01
            gamma_i_minus_2 = 0
            img_ori = x_ori[i_img]
            score_ori = f_x[i_img]
            x_iminus1 = img_ori.clone()
            T_total = 0


            combined_mask = get_mask(img_ori.cpu().numpy())
            jnd_map = torch.from_numpy(compute_jnd(img_ori).astype(np.float32)).to(device)
            image_size = img_ori.shape[-2:]
            dct_mask = get_zig_zag_mask([0, 0.5], image_size).to(device)

            for i in range (30):

                d_tex = get_dtex(img_ori.shape[1:3])
                gamma_i = gamma_i_minus_1 + (gamma_i_minus_1 - gamma_i_minus_2)
                x_i, gamma_i, T_total = Single_Attack(img_ori, score_ori, x_iminus1, gamma_i, gamma_i_minus_1, gamma_i_minus_2, d_tex, combined_mask, jnd_map, T_total, dct_mask)
                x_iminus1 = x_i.clone()
                gamma_i_minus_2 = gamma_i_minus_1
                gamma_i_minus_1 = gamma_i

                if T_total >= Q_max:
                    break

            x_pert_bests[n][i_img] = x_i

    # 50张img
    adv_imgs = x_pert_bests
    return adv_imgs, x_oris

def seed_everything(seed=555):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='TReS')  # 'DBCNN' || 'UNIQUE' || 'TReS' || 'LIQE'
    parser.add_argument('--dataset', type=str, default='CSIQ')  # 'live' || 'csiq' || 'tid'
    parser.add_argument('--seed', type=int, default=555)
    parser.add_argument('--save_dir', type=str, default='./results/')
    parser.add_argument('--log_dir', type=str, default='./logs')

    args = parser.parse_args()


    seed_everything(seed=args.seed)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

   
    args.model == 'DBCNN':
    model = torch.nn.DataParallel(DBCNN()).train(False).to(device)
    ckpt = '' # path to model 
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint)



    dataset = ImageDataset(csv_file='path_to_image', img_dir=dataset_path_dict[args.dataset],
                           transform=test_transform)
    dataset_loader = DataLoader(dataset, batch_size=args.bs)

    n_queries = np.ones(args.n_sample)
    fx = []
    gt = []
    with torch.no_grad():

        adv_imgs, x_oris = attackIQA(dataset_loader=dataset_loader, args=args)

        pred_advs = []
        distortions = []
        RGO = []
        Dist = []

        for n in range(args.n_sample // args.bs):
            # a batch of images: x_oris[n]
            adv_img = adv_imgs[n].clone()
            x_ori = x_oris[n].clone()
            f_x = fx[n]
            pred_adv = predict_quality_score(adv_img, args)
            pred_advs.append(pred_adv.cpu().numpy())
            distortions.append((abs((adv_img - x_ori).max(dim=1)[0].cpu().numpy())).max())
            for idx in range(x_ori.shape[0]):
                RGO.append((abs(pred_adv[idx].cpu().numpy() - f_x[idx])) / max(10.0 - f_x[idx], f_x[idx] - 0.0))


        RGO = np.mean(RGO)
        logger.log('RGO={:.4f}'.format(RGO))
        
        srcc = scipy.stats.mstats.spearmanr(x=pred_advs, y=gt)[0]
        logger.log('srcc_adv={:.4f}'.format(srcc))

        plcc = scipy.stats.mstats.pearsonr(x=pred_advs, y=gt)[0]
        logger.log('plcc_adv={:.4f}'.format(plcc))

        logger.log('Done...')

