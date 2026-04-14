# PhysCoT Remote Training — Usage Guide

8× H100 上跑 Exp B（LLaMA-Factory SFT）和 Exp C（Joint LM + Recon Loss）。两实验共享 base 模型和数据，各约 15h。

- **代码（GitHub）**: https://github.com/zouyueying/PhysCoT-Remote-Training
- **数据/模型（HF，private）**: https://huggingface.co/datasets/yueying-117/PhysCoT-Data

HF repo 是 private，先把 HF 用户名发给 owner 加 collaborator。

## 1. 下载代码

```bash
git clone https://github.com/zouyueying/PhysCoT-Remote-Training.git physcot
cd physcot
```

## 2. 下载数据和模型（约 43 GB）

```bash
pip install -U "huggingface_hub[cli]"
hf auth login   # 用你的 HF token

export DATA_ROOT=/path/to/your/data_root   # 自己选位置，需 50G+ 空闲
mkdir -p $DATA_ROOT && cd $DATA_ROOT

# 一次拿所有数据+模型
hf download yueying-117/PhysCoT-Data --repo-type dataset --local-dir .

# 解压（parsed_frames 和 features 都打成了 tar，避免 HF 25k 文件 commit 上限）
tar -xf parsed_frames.tar && rm parsed_frames.tar
tar -xf features.tar && rm features.tar
```

最终目录结构：
```
$DATA_ROOT/
├── parsed_frames/
├── stage1_merged_fixed/
├── stage1_checkpoint_epoch3/decoder.pt
├── features/{flow,depth,track}/*.pt
└── ladm_physcot.json
```

## 3. 接入代码目录（软链，避免拷贝）

```bash
cd /path/to/physcot
mkdir -p checkpoints stage1
ln -s $DATA_ROOT/stage1_merged_fixed checkpoints/stage1_merged_fixed
ln -s $DATA_ROOT/stage1_checkpoint_epoch3 checkpoints/stage1/checkpoint-epoch3
ln -s $DATA_ROOT/features features
cp $DATA_ROOT/ladm_physcot.json train/data/ladm_physcot.json
```

## 4. 装环境

```bash
bash scripts/setup_env.sh
```

## 5. 关键：remap 数据路径

```bash
bash scripts/remap_paths.sh $DATA_ROOT
# 输出必须显示 "Exists: True"，否则继续排查
```

## 6. 跑实验

```bash
# Exp B: LLaMA-Factory Full FT SFT (~15h, 8x H100)
bash scripts/run_exp_b.sh
# 产出: checkpoints/stage2_full_ft/

# Exp C: Joint LM + Recon Loss (~15h, 核心实验)
bash scripts/run_exp_c.sh
# 产出: checkpoints/stage2_full_ft_joint/
```

日志自动落在 `logs/exp_{b,c}_YYYYMMDD_HHMMSS.log`，tensorboard 在 output_dir 下。

## 7. 评测

```bash
cd eval
python inference.py --index_json /path/to/test_index_local.json \
    --model_path ../checkpoints/stage2_full_ft \
    --model_name PhysCoT-SFT-ExpB --save_dir ../results/exp_b
python eval.py --json_file_path ../results/exp_b/PhysCoT-SFT-ExpB.json
# exp_c 同理
```

## 三条红线

1. **必须先跑 `remap_paths.sh`**：JSON 里的 image 路径默认指向原作者服务器
2. **`model_name_or_path` 必须是 `stage1_merged_fixed`**：已 bake 好 3 个 physics special tokens + 对应 embedding。换成原版 Qwen2.5-VL-7B 会在 Exp C 的 `embedding norm > 0.01` 断言处挂掉
3. **硬件**：ZeRO-2，bs=1 × grad_accum=2，bf16。最低 8× A100-80G；H100 80G 更宽裕。低于 80G 显存：调 `cutoff_len`（B 里 10240 → 6144）或加大 grad_accum

## 结果优先级

B 跑完先回传 ViF-Bench ACC，对 paper 结论分支很关键：

| B 结果 | 结论 | 下一步 |
|---|---|---|
| B > 91% | 物理 token 的**格式**本身就有用 | Paper story: format + CoT 结构 |
| B ≈ 91% | token 中性 | C 的重建损失才是价值点 |
| C > B > 91% | physics 语义加分 | 核心 PhysCoT 贡献成立 |
| C > 91% > B | 语义必要、格式不够 | 最强 paper story |

有任何报错直接贴 log，最可能卡在路径 remap 和 HF collaborator 权限。
