# -----------------------------------------------------------------------------
# Functions for Gaussian Mixture Model based Prompt Method with known K
# -----------------------------------------------------------------------------
import os
import logging
from tqdm import tqdm
import torch
import numpy as np
# from pycave import set_logging_level
from tqdm import auto

from tqdm.auto import tqdm as old_tqdm


import gc

device = torch.device('cuda')
# set_logging_level(logging.WARNING)

class GMMPrompt():
    def __init__(self,num_components=1, train=True):
        super().__init__()
        self.num_components = num_components
        self.num_gmm_samples = 100
        self.covariance_type = "spherical"
        self.convergence_tolerance = 1.0e-7
        self.covariance_regularization = 1.0e-3
        self.topk = 5

        # Initialize GMM model with given number of components and trainer parameters
        if train:
            self.gmm = GMM(
                num_components=int(num_components), 
                trainer_params=dict(gpus=1),
                covariance_type=self.covariance_type,
                convergence_tolerance=self.convergence_tolerance,
                covariance_regularization=self.covariance_regularization,
            )

    def init_gmm_for_eval(self,memory_dir, stage_i):
        self.gmm = GMM.load(os.path.join(memory_dir, f'gmm/gmm_{str(stage_i)}'))

    def fit(self,memory_dir, feats, stage=0):
        
        # feats = torch.cat(all_feats, dim=0)

        if stage > 0:
            prev_samples = np.load(os.path.join(memory_dir, f'gmm/gmm_samples_{stage-1}.npy'))
            prev_samples = torch.from_numpy(prev_samples).to(device)

            # Combine previous samples with current samples
            feats = torch.cat([feats, prev_samples], dim=0)

        # Fit to Gaussian Mixture Model
        self.gmm.fit(feats)
        # del feats

    def predict(self, batch):
        res = dict()
        # Predict batch data to its topk mixture component
        results = self.gmm.predict_proba(batch)                     # Batch, num_components
        _, idx = torch.topk(results, k=self.topk, dim=1)      # Batch, top_k

        idx = idx.squeeze(1)                                        # Batch
        res['batched_prompt'] = self.gmm.model_.means[idx]          # Batch, topk, emb_dim
        res['prompt_idx'] = idx
        res['total_prompt_len'] = res['batched_prompt'].shape[1]

        del results # when NOT using ranodm pick
        gc.collect()
        return res

    def sample(self,memory_dir, stage):
        # For each components in GMM, generate n samples
        samples = self.gmm.sample(self.num_gmm_samples).cpu().numpy() # num_samples, emb_dim

        # Save sample to npy
        np.save(os.path.join(memory_dir, f'gmm/gmm_samples_{stage}.npy'), samples)

        # Save gmm parameters for stage_i
        self.gmm.save(os.path.join(memory_dir, f'gmm/gmm_{stage}'))

        del samples
