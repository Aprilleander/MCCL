# Video-based Generalized Category Discovery via Memory-Guided Consistency-Aware Contrastive Learning

Generalized Category Discovery (GCD) is an emerging and challenging open-world problem that has garnered increasing attention in recent years. The goal of GCD is to categorize all samples in the unlabeled dataset, regardless of whether they belong to known classes or entirely novel ones. Most existing GCD methods focus on discovering categories in static images. However, relying solely on static visual content is often insufficient to reliably discover novel categories. For instance, bird species with highly similar appearances may exhibit distinctly different motion patterns. To bridge this gap, we extend the GCD problem to the video domain and introduce a new setting, termed \textbf{Video-GCD}. Compared with conventional GCD, which primarily focuses on how to leverage unlabeled image data, Video-GCD introduces additional challenges due to complex temporal and spatial dynamics. Thus,  effectively integrating multi-perspective information across time is crucial for accurate Video-GCD. To tackle this challenge, we propose a novel Memory-guided Consistency-aware Contrastive Learning (\textbf{MCCL}) framework, which explicitly captures temporal-spatial cues and incorporates them into contrastive learning through a consistency-guided voting mechanism. MCCL consists of two core components: Consistency-Aware Contrastive Learning (\textbf{CACL}) and Memory-Guided Representation Enhancement (\textbf{MGRE}). AACL exploits multi-perspective temporal features to estimate consistency scores between unlabeled instances, which are then used to weight the contrastive loss accordingly. MGRE introduces a dual-level memory buffer that maintains both feature-level and logit-level representations, providing global context to enhance intra-class compactness and inter-class separability. This in turn refines the consistency estimation in CACL, forming a mutually reinforcing feedback loop between representation learning and consistency modeling. To facilitate a comprehensive evaluation, we construct a new and challenging \textbf{Video-GCD benchmark}, which includes action recognition and bird classification video datasets. Extensive experiments demonstrate that our method significantly outperforms competitive GCD approaches adapted from image-based settings, highlighting the importance of temporal information for discovering novel categories in videos. The code will be publicly available.

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
