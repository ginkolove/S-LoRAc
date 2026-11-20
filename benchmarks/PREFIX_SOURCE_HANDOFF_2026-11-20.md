# Oreo 前缀 KV 来源对照：实验交接（2026-11-20）

分支：`breakdown_hybridpage`。本轮是**服务性能实验**，不是 `accexp` 精度实验，也不是之前的固定 rank、固定 decode 长度或弹性布局实验。接手时先看本文，再看 [全矩阵入口](run_prefix_sources_300s.sh)、[双卡配对入口](run_prefix_sources_after_idle.sh) 和 [单组实际参数](run_lmsys_preemption.sh)。

## 这次要回答什么

同样的长前缀文本，挂不同 LoRA 后会产生不同的 prefix KV。在相同的到达、query、生成长度和 LoRA 分配下，比较五条路径：重新 prefill、从主机读完整 KV（含一个固定缓存对照）、从 MiniKV 完整或逐层重构。完整展开省下后续重构，但占用运行请求也需要的显存；逐层重构省显存，却可能延长每个 decode step。**分别看吞吐、TTFT 和端到端延迟，不只给一个“最快”。**

| 卡及同卡顺序 | 方法名 | miss 怎么办，之后怎么复用 |
|---|---|---|
| GPU0 ① | `active-reprefill` | 用请求对应的 LoRA 对完整前缀做真实 prefill；同 LoRA 只构建一份，不是每个请求各算一遍。 |
| GPU0 ② | `minikv-full_reconstruct` | 从 GPU 静态 BaseKV、该 LoRA 的 MiniKV 及其 B 权重还原全部层。此对照开启 `--reuse-next-reconstructed`，不沿用旧实验的闲置 LRU。 |
| GPU0 ③ | `minikv-layer_serial` | 不保留完整展开前缀；只为当前需要的不同 LoRA 准备单层 workspace，算完一层便覆盖。后续 decode step 再重构；请求私有 KV 照常累积。bounded kernel，1296 workers。 |
| GPU1 ① | `cpu-read` | GPU miss 时，从 CPU 实际传输对应尺寸的完整 K 和 V，不是只传一部分或模拟 sleep。 |
| GPU1 ② | `cpu-read-fixed1-r64` | 同上；额外固定保留 rank=64 组中按**配置概率**最热的那一份（逻辑 ID 2，`dummy-lora-<size>-rank-64-0`），首次用到才加载，不事后用回放计数选。 |

除单层 workspace 外，四种完整前缀路径都按活跃 LoRA 去重，跨到下一 batch 仍有需要就保留，没有需要的普通条目就释放；不回收仍被执行中的请求使用的前缀。只有固定 Top-1 可以额外保留那份闲置热前缀。**没有动态 Top-2、固定 Top-2、并行重构或自适应选层**。两张 GPU 各跑独立 TP=1 服务，不是双卡 TP；同一个模型/长度两边都成功才比较和切换。

## 参数矩阵

K 是十进制千 token。每个长度跑上面全部五种方法，共 11 组、55 场。FP16，目标硬件为两张 A100 PCIe 80GB；使用 `huggyllama/llama-<size>` 对应结构和 tokenizer、`--dummy` 权重。运行真实的模型算子及服务路径，但生成文本内容不用于评估。

| 模型（层数 / hidden） | 固定共享前缀 token | 仅重新 prefill 的池（slots） | 其他四种的池（slots） | 场次 |
|---|---|---:|---:|---:|
| Llama-7B（32 / 4096） | 5000、8000、10000、12000、15000 | 120000 | 125000 | 25 |
| Llama-13B（40 / 5120） | 5000、8000、10000 | 62000 | 65000 | 15 |
| Llama-30B（60 / 6656） | 1000、2000、3000 | 8800 | 8800 | 15 |

这里一个普通 slot 是一个 token 的全层 FP16 K+V，即 `4 × 层数 × hidden` 字节。表中池是总预算，不是在此之外另给 MiniKV 一个同样大的池。7B、13B 的重新 prefill 需要给长前缀 FFN 的临时激活留空间，**只有该方法减物理池**，它的 batch token 上限仍沿用原预算；写论文时要披露这个差异。具体字节参考 [内存预算](llama_memory_budget.sh)，不能把权重加池的字节数当作服务峰值显存。

30B×3000 的容量核算：50 个 LoRA 的 rank 和为 1840；MiniKV 配置静态区约为 `3000 + ceil(3000 × 1840 / 6656) = 3830` slots。加上一份完整展开的 3000，余 1970 给其他用途；**能放一份，不等于能同时放两份，也不是所有 batch 都不会 OOM 的保证**。CPU 读法没有这块静态区。旧版 30B×4K 不是本轮配置。

## 工作负载与测量口径

- 用本地 `/data/hub/LMSYS-33k` 两个回答分支的首个 user/assistant 对作为样本池；按 `data_seed=42` 打乱抽取。query 使用实际 token，decode 严格生成对应回答的 token 数，而不是回放答案文本。公共前缀另行构造，不声称 LMSYS 原始请求本来就有长共享前缀。不是重放完整多轮会话。
- `3 req/s` 齐次泊松到达、持续发送 `300 s`，之后 drain 到完成。到达和适配器抽样 seed=42，模型 seed=42。3 req/s 是到达率，不是完成吞吐，也不保证恰好 900 条。7B 旧 trace 有 878 条，平均 query 57.79、输出 190.61 tokens；8K 点自动得到上下文上限 9529，不按老的 2048/16384 硬截断。
- 50 个逻辑 LoRA，轮转 rank 16/32/64，共 17/17/16 个。**先等概率选 rank 组，再按组内 Zipf(1) 选 adapter**；不是在全部 50 个上做一次 Zipf，也不是旧的 MAF d01 分配。上轮 7B 三组实际请求数为 308/293/277。
- 在线 FCFS continuous batching，`--schedule-interval 1`，每个 decode step 都有准入机会，完成即释放私有 KV，必要时抢占重算。生成终止逻辑知道目标输出数，**调度器不知道每条请求将来要生成多少**，不能按数据集答案长度预留 KV。HTTP 请求携带完整公共 prefix + 用户 query；可复用路径不重新处理已缓存前缀。
- 同一模型/长度的五种方法共同使用一个原子生成的 `trace.json`，比较器校验 SHA-256。不同前缀长度或 tokenizer 对应不同配置，不要求跨点 SHA 相同。迁移后保留数据集和 tokenizer 版本；seed 相同不能替代 SHA 验证。
- CPU 两组的 `--cpu-prefix-pool-gb 100` 是 **100,000,000,000 字节的主机前缀 backing 上限**，不是整机 RSS 上限。每个物理条目占完整前缀字节数，最多分配 50 份；容量放不下 50 份时多个逻辑 LoRA 循环映射同一物理来源，但 GPU 缓存键仍是各自的 LoRA，每个 unique miss 仍搬完整字节。这只适合 dummy 性能对照，不能拿来证明真实模型输出正确。例：7B×5K 实际分 38 份，每次 miss 传 2,621,440,000 字节；H2D 总字节不是实测 PCIe 带宽。
- 主指标 `completed_req_per_s_total` = 全部成功完成数 / 从开始到 drain 结束的壁钟时间；TTFT/E2E 针对全部成功请求。本轮 warmup=0、steady end 截到实际结束，所以 steady 与 total 相同，**不是峰值吞吐**。同时保存 output tok/s、P95、去掉模型权重后的峰值显存和（适用模式的）缓存命中、重建与 H2D 计数。

## 截至交接的实测状态

原机目录：`benchmarks/results/chameleon_r64top1_300s_20261119_150835/`。只把有效结果的摘要写进 Git；原始逐请求记录、trace 和日志仍在原机，没有随代码上传。异机若要复核旧结果，应另行复制该目录。

下表每格为 **全局 req/s / 平均 TTFT 秒 / 平均 E2E 秒**。有数字的场次均为 878/878；来自各方法的正式 `rate3_seed42.json`。

| 方法 | 7B × 5000 | 7B × 8000 |
|---|---|---|
| 重新 prefill | 1.381 / 153.41 / 178.46 | 0.746 / 422.29 / 447.29 |
| CPU 读 | 1.423 / 128.77 / 154.82 | 0.720 / 446.49 / 472.49 |
| CPU 读 + 固定 rank64 Top-1 | 1.387 / 143.34 / 169.98 | 0.740 / 408.91 / 433.74 |
| MiniKV 完整重构 | 1.954 / 58.89 / 76.22 | 请求全部完成，但 reset 断联；无正式结果 |
| MiniKV 逐层串行 | 1.253 / 0.82 / 201.21 | 未启动 |

合计 **8 个正式点 + 1 个回放完成但未确认 + 46 个未启动**。46 个为：7B×8K 逐层 1 个，7B×10K/12K/15K 共 15 个，13B 全部 15 个，30B 全部 15 个。7B×5K 的整组 trace SHA 为 `167b483b48a6d08132542ad5fbdfa6bb7e373d2f003f5cb25b718d9a2bfe7f5f`。旧 80s、固定 rank 或其他日期的实验不要混入本矩阵。

## 收尾故障和有效性

7B×8K 完整重构是在最后的 `/lowra/reset_runtime` 遇到 `ServerDisconnectedError`，并非请求推理失败；仅凭旧日志不能断言断联根因是 keep-alive 或 OOM。已改成：回放成功先原子保存 checkpoint，独立 HTTP 控制连接用同一个 operation ID 有界重试（120 s）；服务端对同 ID 的 reset 去重，客户端确认每个 rank 的所有 slots 都已释放后才发布有效结果。进程退出、泄漏或无法确认仍会失败，不会跳过错误启动下一组。

- 新版成功：`<点>/<方法>/rate3_seed42.json` 有 `result.valid=true`、`reset.status=confirmed`；配对完成另有 `<点>/comparison.json`。旧版已发布结果没有新增的 reset 字段。
- 新版中断：`<方法>/checkpoints/rate3_seed42.json` 的 `replay_valid=true` 只表示请求回放完成，不是正式点；不要手工改名或置 valid。没有 checkpoint 的旧失败点不会自动补出正式结果。
- 整组没比较文件，先看点目录下的 `gpu0.log`、`gpu1.log`，再看方法目录下的 `server.log` 和逐请求记录。比较器应对**单个模型/长度目录**执行，且只在五个正式结果齐全后执行：`"$SLORA_PYTHON" benchmarks/compare_prefix_sources.py "$POINT" --fixed-mode cpu-read-fixed1-r64`。

## 换机和启动

在仓库根目录、Bash 中执行。先切到最新的 `breakdown_hybridpage`。需要两张空闲可见的 80GB GPU、能运行 S-LoRA CUDA 扩展的服务环境、能读 LMSYS 和本地缓存的三个 tokenizer 的 trace 环境，以及高于 100 GB backing 加上其余进程开销的主机可用内存。在异机按实际路径设置这两个变量；数据路径由单组脚本指定。端口 9400/9401、34400/34401 和两段内部端口（从 50000/50016 开始）应可用。

```bash
export SLORA_PYTHON=/home/qlh/anaconda3/envs/slora/bin/python
export SLORA_TRACE_PYTHON=/home/qlh/anaconda3/envs/transformer/bin/python
nvidia-smi -L
bash benchmarks/run_prefix_sources_300s.sh /tmp/oreo-plan --dry-run
```

**方案 A：换机器重新做完整对照（推荐，不混用不同机器的性能）。**

```bash
RUN_DIR="benchmarks/results/chameleon_r64top1_300s_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
nohup bash benchmarks/run_prefix_sources_300s.sh "$RUN_DIR" --start-now > "$RUN_DIR/suite.log" 2>&1 < /dev/null &
echo "PID=$! RUN_DIR=$RUN_DIR"
tail -F "$RUN_DIR/suite.log"
```

**方案 B：保留旧 5K 组，从 7B×8K 开始整组重跑，顺序做完余下十组。** 适合同机续跑；异机使用时标注机器，至少在新机器重复一个共同点检查可比性。不覆盖任何旧结果。二选一，不要与 A 同时启动。

```bash
RUN_DIR="benchmarks/results/chameleon_r64top1_from8k_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
nohup bash -ec '
  out=$1
  for point in 7b:8000 7b:10000 7b:12000 7b:15000 13b:5000 13b:8000 13b:10000 30b:1000 30b:2000 30b:3000; do
    size=${point%:*}
    prefix=${point#*:}
    case "$size" in
      7b) extra=(--reprefill-num-token 120000) ;;
      13b) extra=(--reprefill-num-token 62000) ;;
      30b) extra=() ;;
    esac
    bash benchmarks/run_prefix_sources_after_idle.sh "$out/${size}_p${prefix}" \
      --start-now --model-size "$size" --prefix-len "$prefix" --duration 300 \
      --adapter-profile chameleon --fixed-rank64 --replay-timeout 86400 "${extra[@]}"
  done
' _ "$RUN_DIR" > "$RUN_DIR/suite.log" 2>&1 < /dev/null &
echo "PID=$! RUN_DIR=$RUN_DIR"
tail -F "$RUN_DIR/suite.log"
```

完整矩阵入口不传 `--start-now` 会排队等双卡空闲；单点入口使用 `--queue` 可以排队。没有自动从旧输出目录恢复的功能，不要删锁文件强行并跑。300 秒只是发请求的时间，drain 可能长得多；请求/drain 超时各为 86400 秒。不要因为卡短暂空闲就判定整个矩阵结束，正常完成以 11 个（方案 B 为 10 个）新的 `comparison.json` 为准。

给新会话的第一句话可以是：**“读 benchmarks/PREFIX_SOURCE_HANDOFF_2026-11-20.md，确认 Git 分支和本机结果目录，先报告已完成和待跑场次，再按交接的配置继续，不改 trace、rank 分布和缓存生命周期。”**
