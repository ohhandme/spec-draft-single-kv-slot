# 冻结 Draft 的单 K/V 反馈槽

冻结 Draft，只在第 1 层自注意力里训练一对 K/V。TRUE、MASK、SHUFFLE 共用同一已训练权重，分别读取本题上一轮反馈、屏蔽槽位、读取训练集中另一题同轮深桶的反馈。

## 在独立 A100 上运行

使用 Linux、Python 3.11+ 和与机器 CUDA 驱动匹配的 PyTorch。其余依赖安装命令如下。先在 CPU 夹具上验证入口，再用目标机器的实际权重执行 `smoke` 和 200 题对账。

```bash
python -m pip install -r requirements.txt
python cpu_smoke.py
```

原机的模型目录是 `/data/transfer_from_4090/models`，模型清单是 `/data/task59-a100-closedloop-20261114-v1/models.json`，训练索引来自 `/data/a100-repair-raw5000-20261113-v1/index.jsonl`，RAW 来自 `/data/a100-ultrachat5000-vanilla-20261111-7858dc6a/shards`。开发题单使用已交付的 `historical200_dev_replay_roster.jsonl`，按原字节复制。也可以从原机的 `task62-20261118-cpu-coverage/inputs` 下取得同一个文件。

把以下已核对的输入放在一个挂载目录内。路径可以是软链接。训练索引、模型清单、模型文件和开发题单按 `candidate.json` 对应哈希验证。原 RAW 分片按各分片的收据逐份校验。

```text
inputs/
  models.json
  weights/Qwen3-8B/
  weights/Qwen3-0.6B/
  train/index.jsonl
  raw/shards/<request_id>/trace.json
  raw/shards/<request_id>/receipt.json
  raw/shards/<request_id>/features.safetensors
  rosters/development.jsonl
```

训练使用 RAW 实际起草路径上的教师分布。`collect` 用大模型对完全相同的前缀取分布，再逐分片保存 4096 维 BF16 隐藏向量。全前缀投影和单行投影产生 BF16 舍入差异时，缓存额外保存不同位置的原生 logit 值。训练重建后核对整行 SHA，逐值复现原生 Target 分布。差异稠密时改存完整原生向量。`smoke.json` 记录实际差异项数和编码。缓存大小按实际分片测量。两个实验共用一份教师缓存，同一缓存只允许一个采集者写入。用不同的 `--out` 保存各实验和各损失配置。重新以同一配置执行时，从最近封存的分片、训练断点或已评估请求继续。

在同一台机器同时使用多个实验仓库时，应传入同一个 `--gpu-lock`，由它协调这张卡。第二台独立机器可用默认的本机锁。需要从多台机器共用教师缓存时，挂载目录需支持可靠的文件锁。

支持 `--loss equal` 和 `--loss front`。前者每个有效位置同权，后者从第一草稿位置到第八位置依次用 8 到 1，分母是同一优化步中有效位置的权重和。相同种子、训练数据、初始参数和训练预算用于两种权重，运行目录按损失分开。一次先跑一个配置，检验加权收益时用另一套独立目录重复命令。

各组最终输出和封存 Target token 对齐。`compare` 从逐轮记录复算总接受 token 除以总 Target 验证轮，并给出按题族配对重采样的区间、固定前缀连续接受和计时。历史 200 题用于开发比较。优先先验证实际 A100 上的 BF16 路径，再用与训练和开发按题族隔离的题做确认。

## 启动命令

```bash
set -e
run() {
  python experiment.py "$@" --assets /mnt/spec-inputs --out /srv/spec-runs/kv \
    --teacher-cache /srv/spec-runs/shared-teacher-exact --loss front
}
run smoke
run parity
run collect
run train
run eval --mode raw
run eval --mode true
run eval --mode mask
run eval --mode shuffle
run fixed --mode raw
run fixed --mode true
run fixed --mode mask
run fixed --mode shuffle
run compare
```
