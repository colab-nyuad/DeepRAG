---
license: cc-by-nc-sa-4.0
language:
- en
tags:
- splade
---

## SPLADE-v3

#### SPLADE-v3 is the latest series of SPLADE models.

This checkpoint corresponds to a model that starts from SPLADE++SelfDistil (`naver/splade-cocondenser-selfdistil`), and is
trained with a mix of KL-Div and MarginMSE, with 8 negatives per query sampled from SPLADE++SelfDistil. We used the original MS MARCO
collection **without the titles**.

For more details, see our arXiv companion book: https://arxiv.org/abs/2403.06789  
To use SPLADE, please visit our GitHub repository: https://github.com/naver/splade

## Performance

| | MRR@10 (MS MARCO dev) | avg nDCG@10 (BEIR-13) |
| --- | --- | --- |
| `naver/splade-v3` | 40.2 | 51.7 |

## Citation

If you use our checkpoint, please cite our work:

```
@misc{lassance2024spladev3,
      title={SPLADE-v3: New baselines for SPLADE}, 
      author={Carlos Lassance and Hervé Déjean and Thibault Formal and Stéphane Clinchant},
      year={2024},
      eprint={2403.06789},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```