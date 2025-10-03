# Pytorch
import torch
import torch.nn as nn
# Local
from modules import compress_jpeg, decompress_jpeg
from utils import diff_round, quality_to_factor


class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(height, width, rounding=rounding, factor=factor)

    def forward(self, x):
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered

    def set_quality(self, quality):
        factor = quality_to_factor(quality)
        self.compress.factor = factor
        self.decompress.factor = factor


if __name__ == '__main__':
    with torch.no_grad():
        import cv2
        import numpy as np
        import os
        path = "/data3/risa/lisanqian/lijialin/Code/Natural_contrast/result/clean"
        out_path = "/data3/risa/lisanqian/lijialin/Code/Natural_contrast/result/block/block"
        ps = os.listdir(path)
        jpeg = DiffJPEG(256, 256, differentiable=True)
        quality = 30
        jpeg.set_quality(quality)
        for p in ps:
            img = cv2.imread(os.path.join(path, p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inputs = np.transpose(img, (2, 0, 1)) # 3, 512, 512
            inputs = inputs[np.newaxis, ...] # 1, 3, 512, 512
            tensor = torch.FloatTensor(inputs)
            outputs = jpeg(tensor)
            outputs = outputs.detach().numpy()
            outputs = np.transpose(outputs[0], (1, 2, 0))
            outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_path, p), outputs)
        # print(img.shape)
        # img = cv2.imread("Lena.png")





        # from skimage.metrics import peak_signal_noise_ratio as PSNR
        # img = cv2.imread("Lena.png")
        # print(PSNR(np.uint8(outputs), np.uint8(img)))
