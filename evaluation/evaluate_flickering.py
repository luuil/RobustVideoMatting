"""
LR (Low-Resolution) evaluation.

Note, the script only does evaluation. You will need to first inference yourself and save the results to disk
Expected directory format for both prediction and ground-truth is:

    videomatte_512x288
        ├── videomatte_motion
          ├── pha
            ├── 0000
              ├── 0000.png
          ├── fgr
            ├── 0000
              ├── 0000.png
        ├── videomatte_static
          ├── pha
            ├── 0000
              ├── 0000.png
          ├── fgr
            ├── 0000
              ├── 0000.png

Prediction must have the exact file structure and file name as the ground-truth,
meaning that if the ground-truth is png/jpg, prediction should be png/jpg.

Example usage:

python evaluate.py \
    --pred-dir PATH_TO_PREDICTIONS/videomatte_512x288 \
    --true-dir PATH_TO_GROUNDTURTH/videomatte_512x288
    
An excel sheet with evaluation results will be written to "PATH_TO_PREDICTIONS/videomatte_512x288/videomatte_512x288.xlsx"
"""


import logging
import argparse
import os
import cv2
import numpy as np
import xlsxwriter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pims


class Evaluator:
    def __init__(self):
        self.parse_args()
        self.init_metrics()
        self.evaluate()
        self.write_excel()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--pred-dir', type=str, required=True)
        parser.add_argument('--num-workers', type=int, default=48)
        parser.add_argument('--metrics', type=str, nargs='+', default=['flk1', 'flk2'])
        self.args = parser.parse_args()
        
    def init_metrics(self):
        self.flk1 = MetricFlickering1()
        self.flk2 = MetricFlickering2()
        
    def evaluate(self):
        tasks = []
        position = 0
        target = 'alpha'  # also can be: com, fgr
        
        with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
          if os.path.exists(os.path.join(self.args.pred_dir, target)):
            for video in sorted(os.listdir(os.path.join(self.args.pred_dir, target))):
              future = executor.submit(self.evaluate_worker, target, video, position)
              tasks.append((target, video, future))
              position += 1
                  
        self.results = [(target, video, future.result()) for target, video, future in tasks]
        
    def write_excel(self):
        workbook = xlsxwriter.Workbook(os.path.join(self.args.pred_dir, f'{os.path.basename(self.args.pred_dir)}.flickering1.xlsx'))
        targetsummarysheet = workbook.add_worksheet('summary_target')
        summarysheet = workbook.add_worksheet('summary')
        metricsheets = [workbook.add_worksheet(metric) for metric in self.results[0][2].keys()]
        
        for row, (target, video, metrics) in enumerate(self.results):
            if row == 0:
                header_data = ['target', 'video'] + list(metrics.keys())
                targetsummarysheet.write_row(0, 0, header_data)

            targetsummarysheet.write(row + 1, 0, target)
            targetsummarysheet.write(row + 1, 1, video)
            for col, metric_name in enumerate(metrics.keys()):
                targetsummarysheet.write(row + 1, col + 2, f'=AVERAGE({metric_name}!C{row+3}:BZZ{row+3})')
        
        for i, metric in enumerate(self.results[0][2].keys()):
            summarysheet.write(i, 0, metric)
            summarysheet.write(i, 1, f'={metric}!B2')
                
        for row, (target, video, metrics) in enumerate(self.results):
            for metricsheet, metric in zip(metricsheets, metrics.values()):
                # Write the header
                if row == 0:
                    metricsheet.write(1, 0, 'Average')
                    metricsheet.write(1, 1, f'=AVERAGE(C3:BZZ9999)')
                    for col in range(len(metric)):
                        metricsheet.write(0, col + 2, col)
                        colname = xlsxwriter.utility.xl_col_to_name(col + 2)
                        metricsheet.write(1, col + 2, f'=AVERAGE({colname}3:{colname}9999)')
                        
                metricsheet.write(row + 2, 0, target)
                metricsheet.write(row + 2, 1, video)
                metricsheet.write_row(row + 2, 2, metric)
        
        workbook.close()

    def evaluate_worker(self, target, video, position):
      metrics = {metric_name : [] for metric_name in self.args.metrics}
      reader = pims.PyAVVideoReader(os.path.join(self.args.pred_dir, target, video))
      
      prev = None
      
      for i, frame in enumerate(tqdm(reader[:], desc=f'{target} {video}', position=position, dynamic_ncols=True)):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255

        if 'flk1' in self.args.metrics:
          if i == 0:
            metrics['flk1'].append(0)
          else:
            metrics['flk1'].append(self.flk1(prev, frame))

        if 'flk2' in self.args.metrics:
          if i == 0:
            metrics['flk2'].append(0)
          else:
            metrics['flk2'].append(self.flk2(prev, frame))
                  
        prev = frame

      return metrics

class MetricFlickering1:
    def __call__(self, last, cur):
      flk = last - cur
      flk = np.sum(np.abs(flk)) / last.size
      return flk * 255

class MetricFlickering2:
    def __call__(self, last, cur):
      flk = (last - cur) ** 2
      flk = np.sum(flk) / last.size
      flk = np.sqrt(flk)
      return flk * 255


if __name__ == '__main__':
  import sys
  logging.basicConfig(level=logging.INFO)

  pred_dir = rf'\\HUYA-LUUIL\liulu-results\robust_video_matting\ours_video2_results'
  exps = ['bodyseg_id124_224x160_opt.pb', 'hyseg_body_cloud_landscape_1.2.pb']
  exps = ['qingfeng']
  exps = [os.path.join(pred_dir, e) for e in exps]

  for exp in exps:
    if os.path.exists(exp):
      logging.info(f'eval: {exp}')
      sys.argv = [''] + f'--pred-dir {exp} --num-workers 8 --metrics flk1 flk2'.split()
      Evaluator()