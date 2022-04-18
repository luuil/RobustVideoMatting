"""
python inference_speed_test.py \
    --model-variant mobilenetv3 \
    --resolution 1920 1080 \
    --downsample-ratio 0.25 \
    --precision float32
"""

import argparse
import torch
from tqdm import tqdm
import time
from torch.nn import functional as F
import numpy as np

from model.model_rvm import MattingNetworkRVM
from model.model import MattingNetwork
from inference import auto_downsample_ratio

torch.backends.cudnn.benchmark = True

class InferenceSpeedTest:
    def __init__(self):
        self.parse_args()
        self.init_model()
        self.loop()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-variant', type=str, required=True)
        parser.add_argument('--resolution', type=int, required=True, nargs=2)
        parser.add_argument('--downsample-ratio', type=float, default=1.)
        parser.add_argument('--precision', type=str, default='float32')
        parser.add_argument('--disable-refiner', action='store_true')
        self.args = parser.parse_args()
        
    def init_model(self):
        self.device = 'cuda'
        self.precision = {'float32': torch.float32, 'float16': torch.float16}[self.args.precision]
        self.model = MattingNetwork(self.args.model_variant)

        w, h = (self.args.downsample_ratio * np.asarray(self.args.resolution)).astype(int)
        self.flops(self.model, w, h)
        
        self.model = self.model.to(device=self.device, dtype=self.precision).eval()
        # self.model = torch.jit.script(self.model)
        # self.model = torch.jit.freeze(self.model)
    
    
    
    def loop(self):
        w, h = self.args.resolution
        src = torch.randn((1, 3, h, w), device=self.device, dtype=self.precision)
        with torch.no_grad():
            rec = None, None, None, None
            downsample_ratio = auto_downsample_ratio(h, w) if self.args.downsample_ratio == 1 else self.args.downsample_ratio
            t = 0
            for i in tqdm(range(1000)):
                t0 = time.time()
                # fgr, pha, *rec = self.model(src, *rec, downsample_ratio)
                
                # src_sm = self._interpolate_factor(src, downsample_ratio) if downsample_ratio != 1 else src
                pha = self.model(src, *rec, downsample_ratio=downsample_ratio, segmentation_pass=True)[0]
                # pha = self._interpolate_size(pha, (w, h)) if downsample_ratio != 1 else pha

                torch.cuda.synchronize()
                t1 = time.time()
                if i > 2: t += t1 - t0
                if i%100 == 0 or i < 5: print(f'cur({i})={(t1-t0)*1000}ms')
            print(f'\n{w}x{h}x{downsample_ratio}')
            print(f'avg={1000*t/(i-2)}ms')
    
    @staticmethod
    def flops(net, w, h):
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(net, (3, h, w), as_strings=True, print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    @staticmethod
    def _interpolate_size(x: torch.Tensor, size):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), size=size,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, size=size,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x
    
    @staticmethod
    def _interpolate_factor(x: torch.Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x

if __name__ == '__main__':
  import sys
#   args = '--model-variant mobilenetv3 --resolution 1920 1080 --downsample-ratio 0.25 --precision float32'
#   args = '--model-variant mobilenetv3 --resolution 1280 720 --precision float16'
  args = '--model-variant mobilenetv3 --resolution 1920 1080  --downsample-ratio 0.2 --precision float16'
  sys.argv += args.split()
  InferenceSpeedTest()