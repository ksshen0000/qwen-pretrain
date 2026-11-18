# Qwen 单卡预训练

从零训练 Qwen3-0.6B、延长训练阶段和中断恢复。训练使用 FP32 主权重 + BF16 autocast、
分组 AdamW（矩阵含 embedding 衰减，norm/bias 不衰减，fused）、随机窗口采样、梯度累积及固定验证窗口，
不改变已有数据格式。

旧 checkpoint（BF16 参数、单参数组 AdamW）可直接 resume：权重和 Adam 动量无损升为 FP32，
优化器状态按参数顺序映射到新分组，学习率曲线和步数不变，不会重新升温。代价是显存约多 5GB、
完整 `.pt` 约翻倍；milestone 仍保存为 BF16。显存不足时用 `--param-dtype bfloat16` 回退。
日志和 `metrics.jsonl` 增加 `grad_norm`。

**完整 `.pt` checkpoint 负责训练恢复；milestone `.safetensors` 只负责评估和生成。**
原来的 1B 和 5B 阶段 checkpoint 可以直接使用，无需转换。

## 安装与入口

在原来能正常训练的 Python 环境中，从项目根目录执行：

```bash
python -m qwen_pretrain --help
```

不需要安装项目本身。也可以注册命令入口：

```bash
pip install -e '.[prepare]'
qwen-pretrain --help
```

`prepare` 需要 PyArrow，其余命令需要原来的 PyTorch、Transformers、NumPy 和 safetensors。
依赖版本请沿用服务器上已能正常训练的环境，本次没有重新确定依赖版本组合。

## 三种训练操作

| 命令 | 作用 | 学习率与优化器 |
|---|---|---|
| `train` | 从随机权重开始 | 新建优化器和 warmup + cosine 曲线 |
| `extend` | 从完整 checkpoint 开启新的训练阶段 | 恢复优化器，显式创建新曲线和终点 |
| `resume` | 恢复一个已经建立的阶段 | 继续已保存的优化器、曲线、终点和随机状态 |

首次训练：

```bash
python -m qwen_pretrain train --config configs/train_1b.json
```

从旧的 1B checkpoint 延长到累计 5B，沿用你当前服务器上的数据路径：

```bash
python -m qwen_pretrain extend \
  --config configs/extend_5b.json \
  --checkpoint checkpoints/step_015259.pt \
  --data-dir /root/autodl-tmp/qwen-pretrain/prepared_next4b_parallel \
  --run-dir /root/autodl-tmp/qwen-pretrain/runs/5b
```

该示例从 checkpoint 当前 LR 开始，用 500 个**优化器更新步**升到 `3e-4`，
随后衰减到累计 5B 时的 `1e-5`。batch、seq_len、grad_accum、seed 和模型路径继承旧 checkpoint。
这组超参数来自本项目当前训练方案，不是对所有模型/数据都适用的最优值。

新阶段中断后：

```bash
python -m qwen_pretrain resume --run-dir /root/autodl-tmp/qwen-pretrain/runs/5b
```

`resume` 不提供重新升温或更改终点的参数。需要延长终点时，再用 `extend` 和新的 run 目录。
重复执行 `extend` 指向已有 checkpoint 的输出会报错，避免意外重置 warmup。

**正在运行的旧 5B 阶段不需要重新开启。** 可以等进程停止后，直接恢复原来的平铺目录：

```bash
python -m qwen_pretrain resume \
  --run-dir /root/autodl-tmp/qwen-pretrain/checkpoints_5b
```

它会识别该目录中的 `step_*.pt`，继承原数据和 milestone 路径，继续原曲线。
原 1B 的完整 checkpoint 会保留在原位置。

## 配置规则

- JSON 使用平铺字段，CLI 参数覆盖 JSON，未指定的恢复参数继承 checkpoint。
- token 数支持 `5B`、`250M`、`65536` 等写法，milestone 支持 JSON 数组或逗号分隔。
- JSON 和 CLI 中的相对路径均相对于**当前工作目录**；新保存的路径转为绝对路径，后续恢复不依赖工作目录。
- `--target-tokens` 是累计输入 token 数，向上取整到完整更新步。
  默认 batch 下 5B 终点是 76,294 步、5,000,003,584 tokens。
- 每次打印的 `loss` 是当前更新步内微批次的平均，不是最近 `log_every` 步的平均。
- 恢复时不能修改 batch、seq_len、grad_accum、seed 或梯度检查点设置；模型路径可用指向同一位置的软链接。
- 数据路径/元信息/文件大小变化需显式 `--allow-data-change`。
  检查不包含全部数据内容的哈希；tokenizer 元信息不一致时不会继续。

新 run 目录结构：

```text
runs/5b/
├── checkpoints/       # 最近两个完整 .pt，含优化器、RNG、LR 曲线、累计进度
├── milestones/        # 永久保留模型权重和 JSON，不含优化器
├── run.json           # 最近一次启动的解析后配置、曲线和起始进度
└── metrics.jsonl      # 追加写入的 train/eval 指标
```

建立新 run 时，会在第一次更新前保存初始 checkpoint；warmup 中断也能用 `resume` 恢复。
完整 checkpoint 在定期保存、评估前、milestone 导出前、到达终点和正常停止时保存。
`--keep-checkpoints 2` 只清理该目录中更旧的 `step_*.pt`，不清理 milestone。

Ctrl+C、SIGTERM、SIGHUP 会等待当前完整更新步结束后保存退出。
`kill -9`、断电、OOM、CUDA 错误等从最近成功保存的 checkpoint 恢复。
写入采用临时文件、fsync、原子替换；新文件成功发布后才删除旧 checkpoint。
每个 checkpoint 输出目录有进程锁，防止同目录并发训练写入。
损坏的最新 checkpoint 不会静默回退；可以显式 `--checkpoint` 选择较早文件，使用新的 run 目录保留分支。

## 数据准备

从原始切片准备新数据，或者核对/跳过旧 1B 文档并复制原验证集：

```bash
python -m qwen_pretrain prepare \
  --parquet data/sample/10BT/00[0-9]_00000.parquet \
  --model-dir model \
  --continue-from prepared \
  --additional-train-tokens 4B \
  --output-dir /root/autodl-tmp/qwen-pretrain/prepared_next4b_parallel \
  --workers 12 --batch-rows 128 --inflight 24 --progress-every 10000000
```

旧数据不会修改，旧验证集原样复制。源文档的顺序和 tokenizer 必须与原准备过程一致，
旧二进制逐段核对不一致会报错。文档末尾被旧预算截断的后缀也一起排除，防止同一验证文档进入训练。
源文件内不同位置重复的文本不在此处做内容去重。
`--continue-from` 当前只接受原始数据集，不能以已经续接过的数据集再次续接。

初次准备使用 `--max-tokens`（包含 train 和 val）及 `--val-fraction`；
续接使用 `--additional-train-tokens`（只计算新增 train），并复制旧 val。
输入切片不够时发布实际数量，metadata 中 `preparation_complete` 为 `false` 并打印警告。
数据准备本身尚不支持断点续接；中断后选新输出目录，或显式 `--overwrite` 重建新目录。

## 生成与评估

milestone 可以直接生成，不需要加载优化器：

```bash
python -m qwen_pretrain generate \
  --checkpoint runs/5b/milestones/tokens_2000000000.safetensors \
  --prompt 'The capital of France is' \
  --output logs/generation_2b.txt
```

也支持完整 `.pt`，默认 `--model-dir model`、`--max-new-tokens 80`、GPU 贪心生成。
省略 `--prompt` 使用原来的五个英文提示；可重复传 `--prompt`。
随机生成加 `--sample --temperature 0.8 --top-p 0.9 --seed 42`；CPU 加 `--device cpu`。

固定评估所有 milestone：

```bash
python -m qwen_pretrain evaluate \
  --milestone-dir runs/5b/milestones \
  --data-dir /root/autodl-tmp/qwen-pretrain/prepared_next4b_parallel \
  --output-dir eval_results/5b
```

输出 CSV、Markdown 和生成样本。保持原固定验证窗口、10 道事实题、10 道语法题及五个英文提示。
这些是演示用探针，不代表全面能力评测；训练期间的 val loss 与独立评估的窗口/批数不同，数值不宜直接比较。

查看受信任的完整 checkpoint 的进度和曲线：

```bash
python -m qwen_pretrain inspect checkpoints/step_015259.pt
```

训练恢复/inspect 使用完整 PyTorch 序列化，只加载自己的受信任 checkpoint。
生成读取 `.pt` 时保留 `weights_only=True`，不自动回退到不受限的 pickle。

## 导出为 Hugging Face 格式

后训练（SFT/DPO/GRPO）交给 TRL、LLaMA-Factory 等现成框架；先把 milestone 或 `.pt` 导出成标准模型目录：

```bash
python -m qwen_pretrain export \
  --checkpoint runs/5b/milestones/tokens_2000000000.safetensors \
  --model-dir model \
  --output-dir exported/2b
```

输出 `config.json`、`model.safetensors`、tokenizer 文件（含 Qwen3 chat template）和 `training_info.json`
（来源、step、tokens）。默认 BF16，可用 `--dtype float32`。导出后会重新加载并逐张量核对，
先写到 `<output-dir>.tmp` 再改名，中断不会留下看似完整的目录；目标目录非空时拒绝覆盖。
之后用 `AutoModelForCausalLM.from_pretrained("exported/2b")` 加载。

## 后训练（SFT）

`posttrain/` 下是基于 TRL 的独立脚本，输入为上面导出的 HF 目录，不依赖本包的训练代码：

```bash
pip install trl datasets
export HF_ENDPOINT=https://hf-mirror.com   # 国内服务器下载数据集
python posttrain/sft.py --base exported/5b --output-dir sft/5b --limit 2000   # 试跑
nohup python posttrain/sft.py --base exported/5b --output-dir sft/5b > sft.log 2>&1 &
python posttrain/chat.py sft/5b/final
```

- 默认数据集 `HuggingFaceTB/smol-smoltalk`，1 个 epoch；`--dataset` 也可传含 `messages` 列的本地 .jsonl。
- 使用纯 ChatML 模板（不含 Qwen3 的 `<think>` 块），只对助手回复计算 loss；模板随最终模型保存，`chat.py` 自动一致。
- FP32 主权重 + BF16 autocast；最终模型存为 BF16。中断后重跑同一命令会从最新 checkpoint 继续。
- 显存不足时用 `--batch-size 2 --grad-accum 32`，或 `--max-length 1024`。
- 分词默认 8 个进程并行（`--num-proc`）；单进程处理 46 万条对话约需 35 分钟。
- `chat.py` 中 `/reset` 清空历史，`/quit` 退出。

## 模块结构

```text
qwen_pretrain/
├── cli.py / config.py     # 命令与配置解析
├── training.py           # 初始化、恢复和 Trainer 更新循环
├── checkpoints.py        # 完整训练状态、旧格式兼容、保留策略
├── milestones.py         # 永久模型快照、导出中断修复
├── schedules.py          # 学习率曲线、累计 token/step 换算
├── runtime.py / io.py    # 随机状态、信号、原子写入、锁、指标
├── modeling.py           # 共用模型构建
├── generation.py         # 生成
├── evaluation.py         # 评估和报告
├── export.py             # 导出为 Hugging Face 模型目录
├── benchmarks.py        # 探针评分
├── prompts.py           # 固定探针内容
└── data/
    ├── tokens.py         # 二进制校验、memmap、训练随机采样
    ├── prepare.py        # 准备流水线、预算和输出管理
    ├── replay.py         # 核对并跳过旧数据文档
    ├── tokenize.py       # worker tokenizer 和 Parquet 遍历
    └── prepare_cli.py    # 准备命令参数
```

部署时需要上传整个 `qwen_pretrain/` 包以及使用的配置。
不要在服务器训练进程尚未停止时覆盖它正在使用的代码；本地审查与服务器运行状态独立。

测试（需要 torch，CPU 即可）：

```bash
python -m unittest discover -s tests -t .
```

优化器迁移已在 CPU 上用小模型测试；在服务器真实 checkpoint 上的 GPU 恢复尚未验证，首次 resume 请核对日志。
