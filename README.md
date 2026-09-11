# AVEmo

《基于视听情感一致性的多模态人脸伪造检测方法》
代码包含数据预处理、训练、测试。

## 目录

- `config/`：实验配置。
- `models/`：AVEmo模型和音频、视频backbone。
- `prep/`：FakeAVCeleb、HiFi-AVDF预处理脚本。
- `vis/`：可视化脚本。
- `weights/`：预训练权重放置目录。

## 环境

建议使用Python 3.10，并准备可用的CUDA版PyTorch环境。

```bash
conda create -n avemo python=3.10
conda activate avemo
pip install -r requirements.txt
```

另外需要安装 `ffmpeg`，并保证命令行可以直接调用：

```bash
ffmpeg -version
```

## 权重

请将预训练权重放到以下位置：

```text
weights/marlin_vit_base_ytf.full.pt
weights/audiomae_pretrained.pth
```



## 预处理

生成 FakeAVCeleb 原始划分：

```bash
python prep/fakeav.py --dataset_root <FakeAVCeleb_ROOT> --output_dir <DATA_ROOT>/processed/fakeavceleb --train_ratio 0.7 --seed 42
```

对 CSV 中的视频做切片、采帧和人脸裁剪：

```bash
python prep/preprocess.py \
  --input_csv <DATA_ROOT>/processed/fakeavceleb/training_set.csv \
  --output_csv <DATA_ROOT>/processed/fakeavceleb/processed_training_set.csv \
  --work_dir <DATA_ROOT>/processed/fakeavceleb/preprocess/train \
  --segment_seconds 3.2 \
  --keep_short \
  --num_frames 16 \
  --face_device cuda:0
```

## 训练

先编辑 `config/avemo.py` 中的数据路径、权重路径和训练参数，然后运行：

```bash
python train.py config/avemo.py
```

训练输出默认保存在 `runs/`。交叉操纵实验可使用 `config/avemo_cross_*.py`，消融实验可使用 `config/avemo_ablation.py`。

## 测试

```bash
python test.py \
  --checkpoint runs/<RUN_NAME>/best_FakeAVCeleb.pth \
  --csv <DATA_ROOT>/processed/fakeavceleb/processed_test_set.csv \
  --output_csv runs/test_predictions.csv
```

输出包括逐样本预测 CSV，以及同名 `.metrics.json` 指标文件。

## 可视化

```bash
python vis/emotion_score_distribution.py \
  --checkpoint runs/<RUN_NAME>/best_FakeAVCeleb.pth \
  --processed_test <DATA_ROOT>/processed/fakeavceleb/processed_test_set.csv \
  --test_set <DATA_ROOT>/processed/fakeavceleb/test_set.csv \
  --metadata <FakeAVCeleb_ROOT>/meta_data.csv

python vis/emotion_temporal_curves.py \
  --checkpoint runs/<RUN_NAME>/best_FakeAVCeleb.pth
```

默认可视化结果保存在 `runs/avemo_emotion_vis/`。
