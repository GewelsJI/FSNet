# Evaluation Toolboox for VSOD task

## Introduction

- We directly utilize the benchmark tool from [DAVSOD](https://github.com/DengPingFan/DAVSOD/tree/master/EvaluateTool)
- You can evaluate the model performance (S-measure, E-measure, F-measure and MAE) using the one-key matlab 
  code `main_VSOD.m` in `./FSNet/eval/` directory.


## Related Citations (BibTeX)

If you find this useful, please cite the related works as follows: SSAV model/DAVSOD dataset
```
@InProceedings{Fan_2019_CVPR,
   author = {Fan, Deng-Ping and Wang, Wenguan and Cheng, Ming-Ming and Shen, Jianbing}, 
   title = {Shifting More Attention to Video Salient Object Detection},
   booktitle = {IEEE CVPR},
   year = {2019}
}
```

Metrics
```
@inproceedings{Fan2018Enhanced,
   author={Fan, Deng-Ping and Gong, Cheng and Cao, Yang and Ren, Bo and Cheng, Ming-Ming and Borji, Ali},
   title={{Enhanced-alignment Measure for Binary Foreground Map Evaluation}},
   booktitle={IJCAI},
   pages={698--704},
   year={2018}
}

@inproceedings{fan2017structure,
  author    = {Fan, Deng-Ping and Cheng, Ming-Ming and Liu, Yun and Li, Tao and Borji, Ali},
  title     = {{Structure-measure: A New Way to Evaluate Foreground Maps}},
  booktitle = {IEEE ICCV},
  year      = {2017},
  pages     = {4548-4557}
}
```