from typing import Dict

import os
import os.path as osp

import numpy as np

import torch

class Evaluator:
    def __init__(self):
        '''
            Evaluator for the PCQM4M dataset
            Metric is Mean Absolute Error
        '''
        pass

    def eval(self, input_dict):
        '''
            y_true: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_pred: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_true and y_pred need to be of the same type (either numpy.ndarray or torch.Tensor)
        '''
        assert('y_pred' in input_dict)
        assert('y_true' in input_dict)

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        assert((isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray))
                or
                (isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor)))
        assert(y_true.shape == y_pred.shape)
        # assert(len(y_true.shape) == 1)

        if isinstance(y_true, torch.Tensor):
            return {'mae': torch.mean(torch.abs(y_pred - y_true)).cpu().item()}
        else:
            return {'mae': float(np.mean(np.absolute(y_pred - y_true)))}

    def save_test_submission(self, input_dict: Dict, dir_path: str):
        '''
            save test submission file at dir_path
        '''
        assert('y_pred' in input_dict)

        y_pred = input_dict['y_pred']

        filename = osp.join(dir_path, 'y_pred_pcqm4m')

        assert(isinstance(filename, str))
        assert(isinstance(y_pred, np.ndarray) or isinstance(y_pred, torch.Tensor))

        if not osp.exists(dir_path):
            os.makedirs(dir_path)

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        y_pred = y_pred.astype(np.float32)
        np.savez_compressed(filename, y_pred = y_pred)