# Video-based Generalized Category Discovery via Memory-Guided Consistency-Aware Contrastive Learning (MCCL)

> **Paper:** [arXiv:2509.06306](https://arxiv.org/abs/2509.06306)

MCCL extends Generalized Category Discovery (GCD) to videos. It captures temporal-spatial cues and injects them into contrastive learning via a consistency-guided voting mechanism, with two components: Consistency-Aware Contrastive Learning (CACL) and Memory-Guided Representation Enhancement (MGRE). MCCL significantly outperforms image-GCD baselines on five video benchmarks.

## Dependencies

```bash
pip install -r requirements.txt
pip install einops fvcore iopath scikit-learn scipy pandas tqdm loguru pillow simplejson psutil
```

PyTorch ≥1.10 with CUDA (tested on 1.10 and 2.0).

## Data & Config

1. Set dataset paths in `config.py` (`*_root`, `exp_root`).
2. Dataset-specific settings (frames, lr, pretrained checkpoint, output dir) in `configs/<dataset>/TimeSformer_*_8_224.yaml`.

Five datasets (frames + GCD split files required):

| Dataset | Classes |
|---|---|
| UCF101 | 101 |
| SSv2 | 174 |
| Kinetics-400 | 400 |
| VB100 | 101 |
| IBC127 | 127 |

Frame extraction: `k400_process.py`, `cst_merge.py`.

## Pre-trained Models

Backbone = TimeSformer ViT-B/16. Download `jx_vit_base_p16_224-80ecf9dd.pth` (ImageNet-21K) and the MCCL-L stage-1 checkpoints (**TODO: link**), set via `TIMESFORMER.PRETRAINED_MODEL`.

## Training

```bash
python train_net_2gpus_stedge.py --cfg_file configs/ucf101/TimeSformer_ucf_8_224.yaml --dataset_name ucf101 --sharp_weight 0.5
```

All GPUs are used automatically. Each epoch prints k-means + Hungarian `All/Old/New` accuracy; select the best-`All` checkpoint. Other datasets: change `--cfg_file`/`--dataset_name`.

## Results

See the paper for full per-dataset tables; reproduce with the configs above and `--sharp_weight 0.5`.

## Citing

```bibtex
@article{jing2025video,
  title   = {Video-based Generalized Category Discovery via Memory-Guided Consistency-Aware Contrastive Learning},
  author  = {Jing, Zhang and Nan, Pu and Xiang, Xie Yu and Yanming, Guo and Qianqi, Lu and Shiwei, Zou and Jie, Yan and Yan, Chen},
  journal = {arXiv preprint arXiv:2509.06306},
  year    = {2025}
}
```

## Acknowledgements

Built on [SimGCD](https://github.com/CVMI-Lab/SimGCD) and [TimeSformer](https://github.com/facebookresearch/TimeSformer).

## License

MIT — see [LICENSE](LICENSE).
