# Full-Duplex Strategy for Video Object Segmentation (ICCV, 2021)

[Ge-Peng Ji](https://scholar.google.com/citations?user=oaxKYKUAAAAJ&hl=en), 
[Keren Fu](http://www.kerenfu.top/), 
[Zhe Wu](https://scholar.google.com/citations?hl=en&user=jT1s8GkAAAAJ), 
[Deng-Ping Fan](https://scholar.google.com/citations?hl=en&user=kakwJ5QAAAAJ)*, 
[Jianbing Shen](https://scholar.google.com/citations?hl=en&user=_Q3NTToAAAAJ), &
[Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en&oi=ao)

> The codes and results will be released in the near future due to some restrictions of patent application.

# Introduction

Appearance and motion are two important sources of information in video object segmentation (VOS). Previous methods mainly focus on using simplex solutions, lowering the upper bound of feature collaboration among and across these two cues. In this paper, we study a novel framework, termed the FSNet (Full-duplex Strategy Network), which designs a relational cross-attention module (RCAM) to achieve the bidirectional message propagation across embedding subspaces. 
Furthermore, the bidirectional purification module (BPM) is introduced to update the inconsistent features between the spatial-temporal embeddings, effectively improving the model robustness. By considering the mutual restraint within the full-duplex strategy, our FSNet performs the cross-modal feature-passing (i.e., transmission and receiving) simultaneously before the fusion and decoding stage, making it robust to various challenging scenarios (e.g., motion blur, occlusion) in VOS. Extensive experiments on five popular benchmarks (i.e., DAVIS16, FBMS, MCL, SegTrack-V2, and DAVSOD19) show that our FSNet outperforms other state-of-the-arts for both the VOS and video salient object detection tasks.

# Usage

# Benchmark

## U-VOS task

We use the standard evaluation toolbox from [DAVIS16](https://github.com/davisvideochallenge/davis-matlab/tree/davis-2016). All the pre-computed segmentations are downloaded from this [link](https://davischallenge.org/davis2016/soa_compare.html).

## V-SOD task

We use the standard evaluation toolbox from [DAVSOD benchmark](https://github.com/DengPingFan/DAVSOD).

# Ciatation

    @inproceedings{ji2021FSNet,
      title={Full-Duplex Strategy for Video Object Segmentation},
      author={Ji, Ge-Peng and Fu, Keren and Wu, Zhe and Fan, Deng-Ping and Shen, Jianbing and Shao, Ling},
      booktitle={IEEE ICCV},
      year={2021}
    }

