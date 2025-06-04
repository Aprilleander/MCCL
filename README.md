# Video-based Generalized Category Discovery via Memory-Guided Consistency-Aware Contrastive Learning

We propose a novel Memory-guided Consistency-aware Contrastive Learningframework, which explicitly captures temporal-spatial cues and incorporates them into contrastive learning through a consistency-guided voting mechanism. MCCL consists of two core components: Consistency-Aware Contrastive Learning and Memory-Guided Representation Enhancement . AACL exploits multi-perspective temporal features to estimate consistency scores between unlabeled instances, which are then used to weight the contrastive loss accordingly. MGRE introduces a dual-level memory buffer that maintains both feature-level and logit-level representations, providing global context to enhance intra-class compactness and inter-class separability. This in turn refines the consistency estimation in CACL, forming a mutually reinforcing feedback loop between representation learning and consistency modeling. To facilitate a comprehensive evaluation, we construct a new and challenging, which includes action recognition and bird classification video datasets. Extensive experiments demonstrate that our method significantly outperforms competitive GCD approaches adapted from image-based settings, highlighting the importance of temporal information for discovering novel categories in videos. The code will be publicly available.

## Running

### Dependencies

```
pip install -r requirements.txt
```

### Config

Set paths to datasets and desired log directories in ```config.py```


### Datasets

### Scripts

**Train the model**:

```
bash scripts/run_${DATASET_NAME}.sh
```
## Results

## Citing this work


## Acknowledgements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
