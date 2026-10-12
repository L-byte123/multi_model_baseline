# 原始依据与改动边界

参考官方仓库：[CPJKU/dcase2025_task6_baseline](https://github.com/CPJKU/dcase2025_task6_baseline)。
核对日期：2026-10-12。核对的上游 commit：`c78422fcbed579877919620a30075baccf82bf2b`。

具体参考：

- [`d25_t6/passt.py`](https://github.com/CPJKU/dcase2025_task6_baseline/blob/c78422fcbed579877919620a30075baccf82bf2b/d25_t6/passt.py)：PaSST 型号、stride 与特征输出。
- [`d25_t6/retrieval_module.py`](https://github.com/CPJKU/dcase2025_task6_baseline/blob/c78422fcbed579877919620a30075baccf82bf2b/d25_t6/retrieval_module.py)：两个编码器、首token句子特征、线性投影、L2归一化。
- [`requirements.txt`](https://github.com/CPJKU/dcase2025_task6_baseline/blob/c78422fcbed579877919620a30075baccf82bf2b/requirements.txt)：`hear21passt==0.0.26`、`transformers==4.49.0`。

| 原始系统 | 本项目 |
|---|---|
| 音频与自然语言描述检索 | 振动时频表示与设备工况文本故障分类 |
| 音频Mel STFT，32kHz假设 | 用户信号STFT，显式采样率与Hz频率范围 |
| PaSST `passt_s_p16_s16_128_ap468` | 主配置保留该骨干与16×16 stride |
| RoBERTa-large，支持base | 主配置large，PNG简化配置base |
| 投影到1024维共享空间 | 两路各256维，归一化后拼接 |
| 配对对比学习与检索指标 | 故障交叉熵、Accuracy与macro-F1 |
| Lightning、音频caption数据管线 | 简单PyTorch训练循环与CSV数据接口 |

原始检索 checkpoint 的权重键、投影维度和任务头与本项目不同，不能直接 `load_state_dict`。
本项目 PaSST 初始权重从 hear21passt 下载 AudioSet 模型，文本从 Hugging Face 下载 RoBERTa。
替换Mel为线性STFT会改变频率统计与语义，这是需要实验验证的任务迁移，不是等价前端。
PNG/ResNet18 是额外图像版本，不能表述为原文 PaSST 复现。

本仓库代码独立编写；没有重新发布上游未声明许可证的源码。
第三方库和预训练权重遵循各自许可证；本仓库许可证不覆盖它们。

上游README要求引用其相关论文：

> Paul Primus, Florian Schmid, Gerhard Widmer. Estimated Audio–Caption Correspondences Improve
> Language-Based Audio Retrieval. Proceedings of DCASE 2024 Workshop, pp. 121–125, 2024.

```bibtex
@inproceedings{Primus2024,
  author = {Paul Primus and Florian Schmid and Gerhard Widmer},
  title = {Estimated Audio--Caption Correspondences Improve Language-Based Audio Retrieval},
  booktitle = {Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop},
  year = {2024},
  pages = {121--125}
}
```

PaSST骨干相关论文：Koutini et al., Efficient Training of Audio Transformers with Patchout,
Interspeech 2022, DOI: [10.21437/Interspeech.2022-227](https://doi.org/10.21437/Interspeech.2022-227)。
因此“2025官方基线”不是“所有基础方法都在2025首次提出”。

开发使用了 Transformers procedural skill 来核对模型加载与恢复模式；skill 参考：
[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065)。
