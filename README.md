# SDA-VI-ReID

[![Paper](https://img.shields.io/badge/Paper-Springer-blue)](https://link.springer.com/chapter/10.1007/978-3-031-78354-8_1)
[![Conference Poster](https://img.shields.io/badge/Poster-View-green)](https://drive.google.com/file/d/1t3OPbtc1BhQzlpLUUokwhPDwjXoMnjQJ/view?usp=sharing)

## Overview

This repository contains the PyTorch implementation of SDA-VI-ReID, a novel approach to Visible-Infrared Person Re-Identification (VI-ReID). Our method addresses the challenging problem of matching individuals across visible and infrared image modalities.

## Training and Testing

Sample scripts for training and testing across various datasets are provided in `run.sh`. 

To change the number of target IDs, modify the `--target_ids` argument:

```bash
python train.py --target_ids 10
```

## Acknowledgments

Our code builds upon and extends the following excellent works:

- [MMD-ReID](https://github.com/vcl-iisc/MMD-ReID)
- [AGW](https://github.com/mangye16/Cross-Modal-Re-ID-baseline)

Please refer to the official repositories for additional details on:
- Dataset downloading
- Data preparation
- Baseline hyperparameter selection

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{sahu2024sda,
  title={Supervised Domain Adaptation for Data-Efficient Visible-Infrared Person Re-identification},
  author={Sahu, Mihir and Singh, Arjun and Kolekar, Maheshkumar},
  booktitle={Pattern Recognition},
  year={2025}
}
```
