# 验证记录

日期：2026-10-12。

本地验证环境：Windows、Python 3.10、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、
Transformers 4.49.0、tokenizers 0.21.4、hear21passt 0.0.26、timm 0.9.16。
测试运算在 CPU 上执行，预训练权重下载不作为本次测试前提。

最终单元与集成测试：**13 passed**。覆盖：

1. 已知1500Hz正弦信号的STFT频率峰与Nyquist范围检查。
2. 设备模板保留小数与单位，故障标签和文件路径不进入文本。
3. 同group及同文件不能跨训练、验证、测试集合。
4. 原始信号内容完全相同但文件名不同的跨集合泄漏被拒绝。
5. 时频归一化只来自训练窗口；PNG灰度、量化及频率方向与矩阵一致。
6. 新推理信号复用训练统计；不同STFT设置被拒绝。
7. 数值分支使用训练均值/标准差；未见型号有unknown编码；不一致单位被拒绝。
8. 文本冻结后保持eval，没有骨干梯度，但投影层和信号分支仍可训练。
9. 图像单模态和图像＋数值对照前后向。
10. 实际torchvision ResNet18前向（随机初始化，无下载）。
11. 实际hear21passt PaSST前向及反向，确认patch embedding产生梯度（随机初始化，128×32测试输入）。
12. Accuracy、macro-F1与混淆矩阵的已知值核对。
13. 合成数据＋实际小型RoBERTa结构＋微型CNN两轮训练、最佳模型保存、重载、
    16个测试窗口推理；重载概率与训练期评估一致；无label推理CSV也可用。

离线小型RoBERTa为随机初始化，仅用于验证结构与接口；不是语义预训练模型。
没有报告真实轴承数据精度，也没有完成完整预训练PaSST＋RoBERTa-large的真实数据训练。
没有验证CUDA训练速度、多卡训练或所有外部模型下载镜像。

复查命令：

```bash
python -m pip install -e ".[text,image,passt,test]"
python -m pytest -q
python -m mm_baseline.smoke --output runs/smoke
```

如果仅安装image/text extras，PaSST测试会跳过；完整验证需安装passt extras。
