# MiniKV 精度实验手册

核对日期：2026-11-02（北京时间）。资源位置、样本数和完成状态按本机代码、配置及结果文件核对。
本目录使用 **Transformers + PEFT** 做精度实验，与 S-LoRA 的吞吐量/延迟实验独立。

每组比较三种方法，输入问题、参考答案、输出预算相同：

| 方法 | 固定前缀的处理 | Query 与生成阶段 |
|---|---|---|
| `base` | Base Model prefill 得到 BaseKV，之后复用 | 不使用 LoRA |
| `lora_full` | 挂载 LoRA 后 prefill，复用真实完整 prefix KV | 使用 LoRA |
| `lora_minikv` | 通过 BaseKV、真实 LoRA-A 输出 MiniKV、LoRA-B 和 RoPE 重构 prefix KV | 使用同一个 LoRA |

MiniKV 跨层重构是近似的，不保证与 Full 逐 token 相同。复用的是不可变 prefix 模板，
每批请求的工作 KV 独立，不是 S-LoRA Unified Paging 的物理共享实现。
新单卡入口依次完成 Base、Full、MiniKV，第三阶段独立获取 BaseKV/MiniKV；
早期双副本实验复用前两阶段采集的源状态，具体以归档 `experiment` 为准。

目录：[资源](#模型数据集和-lora) · [最终结果](#最终结果索引) · [运行指令](#运行指令) ·
[指标与产物](#指标与结果文件) · [实现指南](IMPLEMENTATION_GUIDE.md) · [Qwen Alpaca 说明](QWEN_ALPACA.md)

## 模型、数据集和 LoRA

### 基础模型

| 模型 | 本机路径或缓存 ID | 上下文说明 |
|---|---|---|
| 原版 LLaMA-7B | `huggyllama/llama-7b` | 2048 |
| 原版 LLaMA-13B | `huggyllama/llama-13b` | 2048 |
| 原版 LLaMA-30B | `huggyllama/llama-30b` | 2048，hidden size 为 6656 |
| 原版 LLaMA-65B | `/data/hub/Llama-65B` | 2048，使用两卡 TP=2，不是 Llama-2-70B |
| Llama-2-7B | `/data/models_2/llama-2-7b-hf` | 4096 |
| Llama-2-13B | `/data/hub/Llama2-13b-hf` | 4096 |
| Mistral-7B-v0.1 | `/data/hub/mistral-7b-v0.1` | 本地最大位置 32768，实验按滑窗和预算限制到 4096 |
| Qwen2.5-7B-Instruct | `/data/hub/qwen2.5-7b-instruct` | 本地 32768；GSM8K/CNN 实验限制到 4096，Alpaca 到 2048 |

三个 `huggyllama` 模型位于本机 HF 缓存：
`/home/qlh/.cache/huggingface/hub/models--huggyllama--llama-{7b,13b,30b}/`。
精度实验使用真实权重，不使用性能实验中的 dummy 权重。

### 原始数据

| 数据集 | 本机目录 | 评测来源 |
|---|---|---|
| `timdettmers/openassistant-guanaco` | `/data/hub/openassistant-guanaco` | train 选 few-shot，原生 eval 取 query |
| `yahma/alpaca-cleaned` | `/data/hub/alpaca_cleaned` | 固定 few-shot 后，从其余数据取样；无官方 test |
| `HiTZ/alpaca_mt` | `/data/hub/alpaca_mt` | 模板文件；英文和翻译 JSON 另存于下述位置 |
| `abisee/cnn_dailymail`，`3.0.0` | `/data/hub/cnn_dailmail` | train 选 few-shot，test 取正文，highlights 为参考摘要 |
| `ucinlp/drop` | `/data/hub/drop` | train 选 few-shot，validation 取问题 |
| `openai/gsm8k`，`main` | `/data/hub/gsm8k` | train 选 few-shot，test 取问题 |
| CoQA | `/data/hub/coqa` | train 选 few-shot，validation 按对话 turn 评测 |
| `alexfabbri/multi_news` | `/data/hub/multi_news` | train 选 few-shot，test 取完整多篇新闻 |
| `tatsu-lab/alpaca`，原版 Alpaca | `/data/hub/alpaca` | 固定 few-shot 后，从 train 剩余数据取样 |

Alpaca-MT 还依赖 `results/alpaca_mt_en_raw.json`、`results/alpaca_mt_raw/`、
`results/alpaca_mt_en_3shot_all_seed42.jsonl` 和 `/data/hub/alpaca_mt/alpaca_mt.py`。
模板作为数据读取，不执行远程 dataset 脚本。
原始辅助金标准在 `results/coqa_annotations/coqa-dev-v1.0.json` 和
`results/drop_annotations/drop_dataset_dev.json`。

### LoRA 对应关系

`SIZE` 的可选值如下明确列出。Rank 来自本地 `adapter_config.json`，不是 dummy rank。

| 模型 | 数据集 | 当前 LoRA 路径 | Rank |
|---|---|---|---:|
| LLaMA-7B/13B/30B | Guanaco | `/data/hub/guanaco-lora/guanaco-SIZE`，`SIZE=7b/13b/30b` | 64 |
| LLaMA-65B | Guanaco | `/data/hub/guanaco-lora/65b` | 64 |
| LLaMA-7B/13B/30B/65B | Alpaca-Cleaned | `/data/hub/alpaca_cleaned-lora/SIZE`，`SIZE=7b/13b/30b/65b` | 16 |
| LLaMA-7B/13B/30B/65B | Alpaca-MT | `/data/hub/alpaca_mt-lora/SIZE`，`SIZE=7b/13b/30b/65b` | 8 |
| Llama-2-7B/13B | CoQA | `/data/hub/coqa-lora/llama2-SIZE`，`SIZE=7b/13b` | 8 |
| Llama-2-7B/13B | DROP | `/data/hub/drop-lora/llama2-SIZE`，`SIZE=7b/13b` | 8 |
| Llama-2-7B/13B | CNN/DailyMail | `/data/hub/cnn_dailmail-lora/llama2-SIZE`，`SIZE=7b/13b` | 8 |
| Llama-2-7B/13B | Multi-News | `/data/hub/multi_news-lora/llama2-SIZE`，`SIZE=7b/13b` | 8 |
| Llama-2-7B/13B | GSM8K | `/data/hub/gsm8k-lora/llama2-SIZE`，`SIZE=7b/13b` | 64 |
| Mistral-7B | CNN/DailyMail | `/data/hub/cnn_dailmail-lora/mistral-lora` | 8 |
| Mistral-7B | DROP | `/data/hub/drop-lora/mistrial-7b` | 8 |
| Mistral-7B | GSM8K | `/data/hub/gsm8k-lora/mistral-7b` | 8 |
| Qwen2.5-7B-Instruct | GSM8K | `/data/hub/gsm8k-lora/qwen2.5-7b` | 16 |
| Qwen2.5-7B-Instruct | CNN/DailyMail | `/data/hub/cnn_dailmail-lora/qwen2.5-7b` | 64 |
| Qwen2.5-7B-Instruct | 原版 Alpaca | `/data/hub/alpaca-lora` | 8 |

**Mistral 路径变更：** 早期结果的 `config.lora` 写的是各数据集 LoRA 根目录，
现在根目录已变为多模型容器，运行时必须用上表的子目录。
DROP 的 `mistrial-7b` 是当前真实拼写，不能擅自改名。
子目录的 base/rank/alpha/targets/bias 已与历史配置核对，但未独立验证历史权重逐字节身份。

## 最终结果索引

下列共 **8 个基础模型、28 组模型/数据集实验**，均有完整三方法预测。
链接相对 `accexp/`。目录名保留历史原样，不能仅凭名字判断时区。
`smoke`、`prepare_check`、`validation` 及中断目录不作为最终结果。

### 原版 LLaMA

| 模型 | 数据集 | 每方法样本数 | 最终结果文件 |
|---|---|---:|---|
| 7B | Guanaco | 100 | [guanaco_7b_random100_seed42.json](results/guanaco_7b_random100_seed42.json) |
| 13B | Guanaco | 100 | [guanaco_13b_random100_seed42.json](results/guanaco_13b_random100_seed42.json) |
| 30B | Guanaco | 100 | [guanaco_30b_random100_seed42.json](results/guanaco_30b_random100_seed42.json) |
| 65B | Guanaco | 100 | [guanaco_65b_tp2_bs3_20261025_115537/result.json](results/guanaco_65b_tp2_bs3_20261025_115537/result.json) |
| 7B | Alpaca-Cleaned | 1000 | [alpaca_cleaned_random1000_20261021_111818/alpaca_cleaned_7b_random1000_seed42.json](results/alpaca_cleaned_random1000_20261021_111818/alpaca_cleaned_7b_random1000_seed42.json) |
| 13B | Alpaca-Cleaned | 1000 | [alpaca_cleaned_random1000_20261021_111818/alpaca_cleaned_13b_random1000_seed42.json](results/alpaca_cleaned_random1000_20261021_111818/alpaca_cleaned_13b_random1000_seed42.json) |
| 30B | Alpaca-Cleaned | 1000 | [alpaca_cleaned_random1000_20261021_111818/alpaca_cleaned_30b_random1000_seed42.json](results/alpaca_cleaned_random1000_20261021_111818/alpaca_cleaned_30b_random1000_seed42.json) |
| 65B | Alpaca-Cleaned | 200 | [alpaca_cleaned_65b_tp2_bs6_n200_20261026_005313/result.json](results/alpaca_cleaned_65b_tp2_bs6_n200_20261026_005313/result.json) |
| 7B | Alpaca-MT | 2400 | [alpaca_mt_non_en_400_20261021_170830/alpaca_mt_7b_non_en_seed42.json](results/alpaca_mt_non_en_400_20261021_170830/alpaca_mt_7b_non_en_seed42.json) |
| 13B | Alpaca-MT | 2400 | [alpaca_mt_non_en_400_20261021_170830/alpaca_mt_13b_non_en_seed42.json](results/alpaca_mt_non_en_400_20261021_170830/alpaca_mt_13b_non_en_seed42.json) |
| 30B | Alpaca-MT | 2400 | [alpaca_mt_non_en_400_20261021_170830/alpaca_mt_30b_non_en_seed42.json](results/alpaca_mt_non_en_400_20261021_170830/alpaca_mt_30b_non_en_seed42.json) |
| 65B | Alpaca-MT | 300 | [alpaca_mt_65b_pt_es_ca_100_20261026_104002/result.json](results/alpaca_mt_65b_pt_es_ca_100_20261026_104002/result.json) |

MT 的 7B/13B/30B 是 `pt/es/ca/eu/gl/at` 六种语言各 400 条；65B 是 `pt/es/ca` 各 100 条。
同一次 MT 实验的各语言任务按源索引对齐，但大小模型这两套不是完全相同的测试集。
上表 MT JSON 的 `scores` 是**未过滤分数**，之前讨论的过滤后结果在：

- 7B/13B/30B：[turn_boundary_v1/summary.json](results/alpaca_mt_non_en_400_20261021_170830/turn_boundary_v1/summary.json) 的 `models.<结果文件名去后缀>.filtered_scores`，及外层 [summary.json](results/alpaca_mt_non_en_400_20261021_170830/summary.json)。
- 65B：[summary.json](results/alpaca_mt_65b_pt_es_ca_100_20261026_104002/summary.json) 的 `filtered_scores`，及 [turn_boundary_v1/summary.json](results/alpaca_mt_65b_pt_es_ca_100_20261026_104002/turn_boundary_v1/summary.json)。

### Llama-2

| 模型 | 数据集 | 每方法样本数 | 最终结果文件 |
|---|---|---:|---|
| 7B | CoQA | 1000 turns | [llama2_coqa_20261029_161342/7b/result.json](results/llama2_coqa_20261029_161342/7b/result.json) |
| 13B | CoQA | 1000 turns | [llama2_coqa_20261029_161342/13b/result.json](results/llama2_coqa_20261029_161342/13b/result.json) |
| 7B | DROP | 1000 | [llama2_drop_20261029_164247/7b/result.json](results/llama2_drop_20261029_164247/7b/result.json) |
| 13B | DROP | 1000 | [llama2_drop_20261029_164247/13b/result.json](results/llama2_drop_20261029_164247/13b/result.json) |
| 7B | CNN/DailyMail | 1000 | [llama2_cnn_20261030_032236/7b/result.json](results/llama2_cnn_20261030_032236/7b/result.json) |
| 13B | CNN/DailyMail | 1000 | [llama2_cnn_20261030_032236/13b/result.json](results/llama2_cnn_20261030_032236/13b/result.json) |
| 7B | GSM8K | 1000 | [llama2_gsm8k_20261101_090108/7b/result.json](results/llama2_gsm8k_20261101_090108/7b/result.json) |
| 13B | GSM8K | 1000 | [llama2_gsm8k_20261101_090108/13b/result.json](results/llama2_gsm8k_20261101_090108/13b/result.json) |
| 7B | Multi-News，旧协议 | 1000 | [llama2_multi_news_20261030_033914/7b/result.json](results/llama2_multi_news_20261030_033914/7b/result.json) |
| 13B | Multi-News，旧协议 | 1000 | [llama2_multi_news_20261030_033914/13b/result.json](results/llama2_multi_news_20261030_033914/13b/result.json) |

**Multi-News 版本警告：** 已完成的是旧 `legacy` Documents/Summary 输入。
当前默认改成了 `alpaca` 格式，本次未找到新版的完整三方法正式结果，不能用旧结果冒充。
旧 JSONL 不会因代码更新自动变成新版，评分器版本也须单独固定。

### Mistral 和 Qwen

| 模型 | 数据集 | 每方法样本数 | 最终结果文件 |
|---|---|---:|---|
| Mistral-7B | CNN/DailyMail | 1000 | [mistral_cnn_bs20_20261028_035710/result.json](results/mistral_cnn_bs20_20261028_035710/result.json) |
| Mistral-7B | DROP | 1000 | [mistral_drop_bs20_gpu1_20261028_092619/result.json](results/mistral_drop_bs20_gpu1_20261028_092619/result.json) |
| Mistral-7B | GSM8K | 1000 | [mistral_gsm8k_bs20_gpu1_20261028_113832/result.json](results/mistral_gsm8k_bs20_gpu1_20261028_113832/result.json) |
| Qwen2.5-7B-Instruct | GSM8K | 1000 | [qwen25_7b_gsm8k_20261101_125540/result.json](results/qwen25_7b_gsm8k_20261101_125540/result.json) |
| Qwen2.5-7B-Instruct | CNN/DailyMail | 1000 | [qwen25_7b_cnn_gpu1_20261101_133310/result.json](results/qwen25_7b_cnn_gpu1_20261101_133310/result.json) |
| Qwen2.5-7B-Instruct | 原版 Alpaca | 1000 | [qwen25_7b_alpaca_gpu0_20261101_150622/result.json](results/qwen25_7b_alpaca_gpu0_20261101_150622/result.json) |

## 运行指令

### 公共设置

先执行此设置块，再选一组实验执行。**不要把所有命令一起运行**；尤其 TP=2 需要两卡空闲。

```bash
cd /home/qlh/S-LoRAc
PY=/home/qlh/anaconda3/envs/transformer/bin/python
TAG=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
RERUN="accexp/results/rerun_${TAG}"
mkdir -p "$RERUN"
export OMP_NUM_THREADS=1
export TORCHELASTIC_SIGNALS_TO_HANDLE=SIGTERM,SIGINT,SIGQUIT
```

使用现有 `transformer` 环境，不用旧 `slora` 环境。评分依赖在 [requirements.txt](requirements.txt)，
缺失时执行 `"$PY" -m pip install -r accexp/requirements.txt`。
新运行写入 `$RERUN`，不覆盖上述历史结果。单卡改到物理卡 1 时只需更改
`CUDA_VISIBLE_DEVICES=1`，进程内仍用 `--device cuda:0`。

以下尽量复用冻结 JSONL；重新抽样必须保留 `.meta.json`。
`--max-samples 1000` 只是取前 1000 条，不等于随机抽样。
旧代码或协议变化后，不承诺当前重跑与归档数值逐字一致。

### LLaMA-7B/13B/30B：Guanaco

固定 3-shot、同一批随机 100 条、256 输出上限。当前显式 batch=1，早期归档未保存该参数。

```bash
for SIZE in 7b 13b 30b; do
  CUDA_VISIBLE_DEVICES=0 "$PY" -u accexp/eval.py \
    --model "huggyllama/llama-${SIZE}" \
    --lora "/data/hub/guanaco-lora/guanaco-${SIZE}" \
    --data accexp/results/guanaco_local_3shot_random100_seed42.jsonl \
    --device cuda:0 --batch-size 1 --weight-format bin --dtype float16 \
    --max-new-tokens 256 --max-samples 0 --seed 42 --local-files-only \
    --output "$RERUN/guanaco_${SIZE}.json" || break
done
```

如需重建，`data/guanaco.py` 可通过 `--train-file` 和 `--test-file` 读取
`/data/hub/openassistant-guanaco/openassistant_best_replies_{train,eval}.jsonl`；
上述 100 条冻结输入才是历史结果使用的准确样本集合。

### LLaMA-7B/13B/30B：Alpaca-Cleaned

复用相同的 1000 条，双卡分别服务 Base/Full，空闲后共同分摊 MiniKV。
这是**两份副本，不是 TP**，batch 依次为 8/4/2。

```bash
for SIZE in 7b 13b 30b; do
  case "$SIZE" in
    7b) BATCH=8 ;;
    13b) BATCH=4 ;;
    30b) BATCH=2 ;;
  esac
  CUDA_VISIBLE_DEVICES=0,1 "$PY" -u accexp/eval.py \
    --model "huggyllama/llama-${SIZE}" \
    --lora "/data/hub/alpaca_cleaned-lora/${SIZE}" \
    --data accexp/results/alpaca_cleaned_random1000_20261021_111818/alpaca_cleaned_random1000_seed42.jsonl \
    --device cuda:0 --lora-device cuda:1 \
    --batch-size "$BATCH" --minikv-batch-size "$BATCH" \
    --weight-format bin --dtype float16 --max-new-tokens 256 \
    --max-samples 0 --seed 42 --local-files-only \
    --output "$RERUN/alpaca_cleaned_${SIZE}.json" || break
done
```

重新抽样并管理三模型连续运行可用 `run_alpaca_cleaned.py --sample-size 1000 --seed 42`，
它依赖已有 `results/alpaca_cleaned_3shot_all_seed42.jsonl`，不是直接接收数据目录的入口。

### LLaMA-7B/13B/30B：Alpaca-MT

六种语言各 400 条，使用两张物理卡 0、1；自动保存 raw 和过滤结果。

```bash
CUDA_VISIBLE_DEVICES=0,1 "$PY" -u accexp/run_alpaca_mt.py \
  --data accexp/results/alpaca_mt_non_en_400_20261021_170830/alpaca_mt_non_en_english_prefix_400_seed42.jsonl \
  --sizes 7b 13b 30b --output-dir "$RERUN/alpaca_mt"
```

重建用 `data/prepare_alpaca_mt.py --samples-per-language 400 --seed 42 --output <新文件>`；
该脚本默认 200，不能省略 400 后仍宣称使用现有的 2400 条设置。

### LLaMA-65B：Guanaco，TP=2

两个 rank 共同执行当前阶段，依次完成 Base、Full、MiniKV。

```bash
mkdir -p "$RERUN/guanaco_65b"
CUDA_VISIBLE_DEVICES=0,1 nohup "$PY" -u \
  -m torch.distributed.run --standalone --nproc_per_node=2 accexp/eval_tp.py \
  --model /data/hub/Llama-65B --lora /data/hub/guanaco-lora/65b \
  --data accexp/results/guanaco_local_3shot_random100_seed42.jsonl \
  --model-weight-format safetensors --adapter-weight-format auto \
  --dtype float16 --batch-size 3 --max-new-tokens 256 --seed 42 \
  --output "$RERUN/guanaco_65b/result.json" \
  > "$RERUN/guanaco_65b/run.log" 2>&1 < /dev/null &
```

### LLaMA-65B：Alpaca-Cleaned，TP=2

保持已完成版本的 200 条、batch=6。启动器自己调用 torchrun，外层不要再套。
先加载模型再处理数据，默认要求两卡各至少 76 GiB 空闲。

```bash
mkdir -p "$RERUN/alpaca_cleaned_65b"
CUDA_VISIBLE_DEVICES=0,1 nohup "$PY" -u accexp/run_alpaca_cleaned_65b.py \
  --model /data/hub/Llama-65B --lora /data/hub/alpaca_cleaned-lora/65b \
  --dataset-dir /data/hub/alpaca_cleaned \
  --sample-size 200 --num-shots 3 --prefix-max-tokens 1024 \
  --batch-size 6 --max-new-tokens 256 --seed 42 \
  --model-weight-format safetensors --dtype float16 \
  --output-dir "$RERUN/alpaca_cleaned_65b" \
  > "$RERUN/alpaca_cleaned_65b/run.log" 2>&1 < /dev/null &
```

### LLaMA-65B：Alpaca-MT，TP=2

三个语言各 100 条、复用历史输入、自动过滤。需要两卡空闲。

```bash
mkdir -p "$RERUN/alpaca_mt_65b"
CUDA_VISIBLE_DEVICES=0,1 nohup "$PY" -u accexp/run_alpaca_mt_65b.py \
  --model /data/hub/Llama-65B --lora /data/hub/alpaca_mt-lora/65b \
  --dataset-dir /data/hub/alpaca_mt \
  --data accexp/results/alpaca_mt_65b_pt_es_ca_100_20261026_104002/alpaca_mt_pt_es_ca_100_seed42.jsonl \
  --languages pt es ca --samples-per-language 100 \
  --batch-size 6 --max-new-tokens 256 --seed 42 --dtype float16 \
  --output-dir "$RERUN/alpaca_mt_65b" \
  > "$RERUN/alpaca_mt_65b/run.log" 2>&1 < /dev/null &
```

### Llama-2-7B/13B：CoQA

两模型依次在单卡执行；runner 自动选择模型、适配器、格式及 batch。
CoQA 的 batch 为 20/8，输出上限为 64，不是 256。

```bash
for SIZE in 7b 13b; do
  CUDA_VISIBLE_DEVICES=0 "$PY" -u accexp/run_llama2_coqa.py \
    --size "$SIZE" \
    --data accexp/results/coqa_llama2_3shot_n1000_seed42_v1.jsonl \
    --output-dir "$RERUN/llama2_coqa/$SIZE" || break
done
```

### Llama-2-7B/13B：DROP

```bash
for SIZE in 7b 13b; do
  CUDA_VISIBLE_DEVICES=0 "$PY" -u accexp/run_llama2_drop.py \
    --size "$SIZE" \
    --data accexp/results/drop_llama2_3shot_n1000_seed42_v1.jsonl \
    --output-dir "$RERUN/llama2_drop/$SIZE" || break
done
```

### Llama-2-7B/13B：CNN/DailyMail

```bash
for SIZE in 7b 13b; do
  CUDA_VISIBLE_DEVICES=0 "$PY" -u accexp/run_llama2_cnn.py \
    --size "$SIZE" \
    --data accexp/results/cnn_llama2_3shot_n1000_seed42_v1.jsonl \
    --output-dir "$RERUN/llama2_cnn/$SIZE" || break
done
```

### Llama-2-7B/13B：GSM8K

```bash
for SIZE in 7b 13b; do
  CUDA_VISIBLE_DEVICES=0 "$PY" -u accexp/run_llama2_gsm8k.py \
    --size "$SIZE" \
    --data accexp/results/gsm8k_llama2_lema_3shot_n1000_seed42_v1.jsonl \
    --output-dir "$RERUN/llama2_gsm8k/$SIZE" || break
done
```

DROP/CNN/GSM8K 的 Llama-2 batch 为 8/4、FP16、输出上限 256。
四个任务都复用历史冻结文件；重建分别用 `data/coqa.py`、`data/llama2_drop.py`、
`data/llama2_cnn.py`、`data/llama2_gsm8k.py`，指定新 `--output`。
两个模型应使用同一份经过双方 tokenizer 长度检查的 JSONL。

### Llama-2-7B/13B：Multi-News

**以下为新版 Alpaca 提示协议的新实验，不是现有 legacy 数值的复现命令。**
原始结果用的是 `results/multi_news_llama2_1shot_n1000_seed42_v1.jsonl`；
要重跑旧输入可传该文件并设 `--require-prompt-format legacy`，但当前评分协议已更新。

```bash
"$PY" -u accexp/data/multi_news.py \
  --prompt-format alpaca --sample-size 1000 --num-shots 1 \
  --prefix-max-tokens 1024 --max-new-tokens 512 --seed 42 \
  --output "$RERUN/multi_news_alpaca_v2.jsonl"

for SIZE in 7b 13b; do
  CUDA_VISIBLE_DEVICES=0 "$PY" -u accexp/run_llama2_multi_news.py \
    --size "$SIZE" --require-prompt-format alpaca \
    --data "$RERUN/multi_news_alpaca_v2.jsonl" \
    --output-dir "$RERUN/llama2_multi_news_alpaca/$SIZE" || break
done
```

### Mistral-7B：CNN/DailyMail

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -u accexp/run_mistral_cnn.py \
  --model /data/hub/mistral-7b-v0.1 \
  --lora /data/hub/cnn_dailmail-lora/mistral-lora --dataset-dir /data/hub/cnn_dailmail \
  --sample-size 1000 --num-shots 3 --prefix-max-tokens 1536 \
  --max-context-len 4096 --max-new-tokens 256 --batch-size 20 \
  --dtype float16 --weight-format bin --seed 42 --output-dir "$RERUN/mistral_cnn"
```

### Mistral-7B：DROP

```bash
CUDA_VISIBLE_DEVICES=1 "$PY" -u accexp/run_mistral_drop.py \
  --model /data/hub/mistral-7b-v0.1 \
  --lora /data/hub/drop-lora/mistrial-7b --dataset-dir /data/hub/drop \
  --gold-file accexp/results/drop_annotations/drop_dataset_dev.json \
  --sample-size 1000 --num-shots 3 --prefix-max-tokens 1536 \
  --max-context-len 4096 --max-new-tokens 256 --batch-size 20 \
  --dtype float16 --weight-format bin --seed 42 --output-dir "$RERUN/mistral_drop"
```

### Mistral-7B：GSM8K

```bash
CUDA_VISIBLE_DEVICES=1 "$PY" -u accexp/run_mistral_gsm8k.py \
  --model /data/hub/mistral-7b-v0.1 \
  --lora /data/hub/gsm8k-lora/mistral-7b --dataset-dir /data/hub/gsm8k \
  --sample-size 1000 --num-shots 3 --prefix-max-tokens 1536 \
  --max-context-len 4096 --max-new-tokens 256 --batch-size 20 \
  --dtype float16 --weight-format bin --seed 42 --output-dir "$RERUN/mistral_gsm8k"
```

### Qwen2.5-7B-Instruct：GSM8K

Qwen 使用原生 Transformers `generate`、默认 FP32，保留严格 KV 一致性检查。
GSM8K 使用基础 Qwen chat template 和最终数值答案提取，不用 Alpaca 模板。

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -u accexp/run_qwen_gsm8k.py \
  --model /data/hub/qwen2.5-7b-instruct \
  --lora /data/hub/gsm8k-lora/qwen2.5-7b --dataset-dir /data/hub/gsm8k \
  --sample-size 1000 --num-shots 3 --prefix-max-tokens 1536 \
  --max-context-len 4096 --max-new-tokens 256 \
  --batch-size 20 --dtype float32 --device cuda:0 --seed 42 --output-dir "$RERUN/qwen_gsm8k"
```

### Qwen2.5-7B-Instruct：CNN/DailyMail

batch=8。适配器 tokenizer 的聊天模板与基础模型不同，已明确弃用其模板；
三方法统一用基础 Qwen 模板，结果保存相关警告，不声称复现作者训练提示。

```bash
CUDA_VISIBLE_DEVICES=1 "$PY" -u accexp/run_qwen_cnn.py \
  --model /data/hub/qwen2.5-7b-instruct \
  --lora /data/hub/cnn_dailmail-lora/qwen2.5-7b --dataset-dir /data/hub/cnn_dailmail \
  --sample-size 1000 --num-shots 3 --prefix-max-tokens 1536 \
  --max-context-len 4096 --max-new-tokens 256 \
  --batch-size 8 --dtype float32 --device cuda:0 --seed 42 --output-dir "$RERUN/qwen_cnn"
```

### Qwen2.5-7B-Instruct：原版 Alpaca

按适配器发布的 `type: alpaca` 配置使用 Alpaca 指令格式，不套 chat template。
当前 seed=42、3-shot 的实际缓存前缀为 167 tokens，输入为 196--318 tokens。
这是当前样本测量值，不是每条 query 固定长度。

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -u accexp/run_qwen_alpaca.py \
  --model /data/hub/qwen2.5-7b-instruct \
  --lora /data/hub/alpaca-lora --dataset-dir /data/hub/alpaca \
  --sample-size 1000 --num-shots 3 --prefix-max-tokens 1024 \
  --max-context-len 2048 --max-new-tokens 256 \
  --batch-size 20 --dtype float32 --device cuda:0 --seed 42 --output-dir "$RERUN/qwen_alpaca"
```

### 后台运行和进度

普通命令为前台版本。nohup 时提前创建目录，加重定向和 `&`，例如：

```bash
RUN_DIR="$RERUN/qwen_alpaca_background"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=0 nohup "$PY" -u accexp/run_qwen_alpaca.py \
  --model /data/hub/qwen2.5-7b-instruct \
  --lora /data/hub/alpaca-lora --dataset-dir /data/hub/alpaca \
  --sample-size 1000 --num-shots 3 --max-new-tokens 256 \
  --batch-size 20 --dtype float32 --seed 42 \
  --output-dir "$RUN_DIR" > "$RUN_DIR/run.log" 2>&1 < /dev/null &

tail -f "$RUN_DIR/run.log"
```

不要同时运行这个示例和上一节的同任务命令。整套循环需要后台执行时，应将完整循环放入
shell 脚本再 nohup，不是只给最后一条命令加 `&`。

## 指标与结果文件

### 指标口径

分数一般以 `[0, 1]` 保存，乘 100 才是百分数。不同任务的 F1 不应当视为同一指标。

| 任务 | 字段 | 说明 |
|---|---|---|
| Guanaco、Alpaca-Cleaned | `exact_match`、`generic_token_f1` | 英文归一化 EM、空格分词重合 F1，不是语义正确率 |
| Alpaca-MT | `exact_match`、`generic_token_f1`、`by_language` | Unicode/多语言归一化及语言宏平均，不同于英文 scorer |
| CoQA | `exact_match`、`coqa_f1` | 多标注者 leave-one-out；样本按 turn，不按 conversation |
| DROP | `exact_match`、`drop_f1` | 数值归一化、分组金标准、多 span 最优匹配 |
| GSM8K | `exact_match`、`flexible_exact_match` | 最终数值严格匹配；无明确答案声明时宽松版本允许末尾数字回退 |
| CNN/DailyMail、Multi-News | `rouge1`、`rouge2`、`rougeL` | 开启词干处理的宏平均 ROUGE F1，主指标 ROUGE-L |
| Qwen 原版 Alpaca | `exact_match`、`generic_token_f1`、`rougeL` | 主指标 Token F1，另存 EM/ROUGE-L 和原始分数 |

EM 要求归一化后整段一致；Token F1 不考虑词序；ROUGE-L 考虑最长公共子序列的相对顺序。
Token F1 中的词不是模型 tokenizer 的 token，F1=0.48 也不意味着 48% 的问题答对。
各 GSM8K 入口接受的答案标记并非完全相同，严格/宽松 EM 不应混列。

### 过滤版本

- Guanaco、LLaMA Alpaca-Cleaned 的归档主要是整段输出分数。
- MT 的过滤分数在额外 summary/`turn_boundary_v1` 中，原始预测不覆盖。
- Mistral DROP/CNN 归档评分完整输出，Llama-2 DROP/CNN 使用明确轮次边界过滤，应注明协议差异。
- Qwen Alpaca 的 `scores` 已过滤，`unfiltered_scores` 为原始分数，三方法使用同一规则。
- 过滤不搜索正确答案、不利用参考文本；合法回答中的保留标题仍可能被误切，应查看 audit/原文。

### 常用文件

| 文件 | 内容 |
|---|---|
| `result.json` 或历史具名 `.json` | 配置、三方法分数、完整预测、KV 检查等 |
| `result.partial.json` 或同名 `.partial.json` | 已完成批次的累计分数，不是最终结果 |
| `result.predictions.jsonl` 或同名 `.predictions.jsonl` | 按批落盘的预测日志 |
| `samples.jsonl` 或具名输入 JSONL | 三方法共享的 `id/prefix/query/references/metadata` |
| 输入 `.meta.json` | 前缀、抽样索引、样本 ID、token 长度、哈希、排除计数 |
| `status.json` | launcher 完成/失败状态；早期直接 `eval.py` 的运行可能没有 |
| `summary.json`、`turn_boundary_v1/` | 多模型/多语言汇总、过滤前后分数与审计 |

判断完成需同时核对最终文件、预期样本数和每条的三种 `predictions`，不能只看进度条或文件夹存在。
发现中断不要覆盖旧记录；这些入口不承诺自动续跑。

## 实验边界

- 固定前缀包含 train 的完整 few-shot，不含当前参考答案。CoQA/DROP/CNN/Multi-News 的当前证据仍在 query，
  因而这不是任意长证据文档被近似后的质量实验。
- `prefix-max-tokens` 是上限而非目标；query 随样本变化，输出可遇 EOS 提前结束。
  精度实验不使用性能实验的固定 `1920+64` 设置。
- Alpaca-Cleaned、原版 Alpaca、Alpaca-MT 有同源关系，MT 各语言也是翻译，不能计作多个独立任务来源。
  排除 few-shot 不代表排除了公开 LoRA 的训练样本；Qwen Alpaca 发布配置的验证集比例为 0。
- 公共 LoRA 的 base 声明可能是其他原始 LLaMA 转换目录；形状兼容不等于独立证明权重身份完全相同。
- 精确复现需固定输入 JSONL、checkpoint、代码、库和评分器版本。以归档
  `config/versions/experiment/scorer_metadata` 为准，不只依赖当前默认参数。
- 本次仅整理文档，不修改评测代码或结果，不启动新的 GPU 实验。
