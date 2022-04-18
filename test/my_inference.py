import os
import datetime
import logging
import torch
import sys
sys.path.append('.')

from inference import convert_video
from model import MattingNetwork


def main_test(exp: str, stage: str, size, segpass=True):
  input_dir = r"\\huya-luuil\liulu-results\robust_video_matting\video\ph"
  # input_dir = r"\\HUYA-LUUIL\liulu-results\seg_stability\test_videos"
  # input_dir = r"\\HUYA-LUUIL\liulu-results\videos"
  # today = datetime.date.today().strftime("%Y%m%d")
  if exp == 'rvm_mobilenetv3':
    out_dir = fr'\\HUYA-LUUIL\liulu-results\robust_video_matting\rvm_video_results\20220107\{exp}_stage4_mbv3'
  elif exp == 'rvm_resnet50':
    out_dir = fr'\\HUYA-LUUIL\liulu-results\robust_video_matting\rvm_video_results\20220107\{exp}_stage4_res50'
  else:
    out_dir = fr'\\HUYA-LUUIL\liulu-results\robust_video_matting\rvm_video_results\20220107\{exp}_{stage}_mbv3'

  out_dir = out_dir if size is None else out_dir + f'_{size[0]}x{size[1]}'
  out_dir = out_dir if segpass else out_dir + '_matte'
  if os.path.exists(out_dir):
    logging.info(f'skip generated: {out_dir}')
    return
  
  ## load model
  seq_chunk = 16
  if exp == 'rvm_mobilenetv3':
    model = MattingNetwork(variant='mobilenetv3').eval().cuda()
    model.load_state_dict(torch.load('test/rvm_mobilenetv3/rvm_mobilenetv3.pth'))
  elif exp == 'rvm_resnet50':
    seq_chunk = 4
    model = MattingNetwork(variant='resnet50').eval().cuda()
    model.load_state_dict(torch.load('test/rvm_resnet50.pth'))
  else:
    if stage == 'stage1':
      ckpt = 'epoch-19.pth'
    elif stage == 'stage2':
      ckpt = 'epoch-21.pth'
    elif stage == 'stage3':
      ckpt = 'epoch-4.pth'
    elif stage == 'stage4':
      ckpt = 'epoch-27.pth'
    else:
      raise NotImplemented()

    weight = f'test/{exp}/{stage}/' + ckpt
    if not os.path.exists(weight):
      logging.info(f'skip not exists weigt: {weight}')
      return

    model = MattingNetwork(variant='mobilenetv3').eval().cuda() # 或 variant="resnet50"
    model.load_state_dict(torch.load(weight))
  
  ## run video
  for v in sorted(os.listdir(input_dir)):
    input_video = os.path.join(input_dir, v)
    output_com = os.path.join(out_dir, 'composite', v)
    output_pha = os.path.join(out_dir, 'pha', v)
    output_fgr = None if segpass else os.path.join(out_dir, 'fgr', v)

    os.makedirs(os.path.dirname(output_com), exist_ok=True)
    os.makedirs(os.path.dirname(output_pha), exist_ok=True)
    if not segpass:
      os.makedirs(os.path.dirname(output_fgr), exist_ok=True)

    convert_video(
      model,                           # The loaded model, can be on any device (cpu or cuda).
      input_source=input_video,        # A video file or an image sequence directory.
      input_resize=size,               # [Optional] Resize the input (also the output).
      downsample_ratio=1,              # [Optional] If None, make downsampled max size be 512px.
      output_type='video',             # Choose "video" or "png_sequence"
      output_composition=output_com,   # File path if video; directory path if png sequence.
      output_alpha=output_pha,         # [Optional] Output the raw alpha prediction.
      output_foreground=output_fgr,    # [Optional] Output the raw foreground prediction.
      output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
      seq_chunk=seq_chunk,             # Process n frames at once for better parallelism.
      num_workers=0,                   # Only for image sequence input. Reader threads.
      progress=True,                   # Print conversion progress.
      segmentation=segpass
    )


if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)

  # inference_size = 288, 512
  inference_size = 160, 224
  # inference_size = None

  exps = ['rvm_mobilenetv3', 'baseline', 'finetune', 'finetune1', 'finetune2']
  exps += [f'exp{idx}' for idx in range(1, 10)]
  exps = ['rvm_mobilenetv3', 'exp9'] + [f'exp{idx}' for idx in range(7, 10)]
  
  stage = 'stage4'

  segpass = True

  for exp in exps:
    main_test(exp, stage, inference_size, segpass=segpass)