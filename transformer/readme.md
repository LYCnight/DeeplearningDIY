## 项目结构
.
├── readme.md
├── self_attention.py       # 单头自注意力
├── train_and_infer.py      # 训练和推理
├── transformer.py          # Transformer模型从零实现
└── utils.py                # 输入与输出转换的工具类

## 使用方法
```
conda activate marker
python train_and_infer.py
```

## requirements
python 3.11.9
pytorch 2.2.0

```
# CUDA 11.8
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```