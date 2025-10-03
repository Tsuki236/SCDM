import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

def save_gray_image(image, path):
    image = image.to("cpu")
    image = (image - torch.min(image)) / (torch.max(image) - torch.min(image)) * 255
    image = np.asarray(image)
    image = image.astype(np.uint8)
    cv2.imwrite(path, image)

def PSNR(img1, img2):
    img1 = img1 / np.max(img1)
    img2 = img2 / np.max(img2)
    psnr = cv2.PSNR(img1, img2, 1.)
    return psnr

def SSIM(img1, img2):
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        img1 = np.reshape(img1, (img1.shape[0], img1.shape[1]))
        img2 = np.reshape(img2, (img1.shape[0], img1.shape[1]))

    img1 = ((img1 - np.min(img1)) / (np.max(img1) - np.min(img1)) * 255).astype(np.uint8)
    img2 = ((img2 - np.min(img2)) / (np.max(img2) - np.min(img2)) * 255).astype(np.uint8)
    return ssim(img1, img2)

def calculate_cdf(hist):
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # 归一化
    return cdf_normalized

def histogram_matching(source, template):
    # 计算源图像和模板图像的直方图
    source_hist, bins = np.histogram(source.flatten(), 256, (0, 256))
    template_hist, bins = np.histogram(template.flatten(), 256, (0, 256))

    # 计算源图像和模板图像的累积分布函数（CDF）
    source_cdf = calculate_cdf(source_hist)
    template_cdf = calculate_cdf(template_hist)

    # 建立映射表
    mapping = np.zeros(256)
    for i in range(256):
        diff = np.abs(source_cdf[i] - template_cdf)
        closest_idx = np.argmin(diff)
        mapping[i] = closest_idx

    # 应用映射到源图像
    matched = cv2.LUT(source, mapping.astype(np.uint8))

    return matched