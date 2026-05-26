# RCCL Primitive Profiler

## 编译

```bash
# 在 projects/rccl 下
./install.sh -f --prefix $PWD/install

# 在 projects/rccl-tests 下
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=$(readlink -f ../../rccl/install) \
      -DBUILD_LOCAL_GPU_TARGET_ONLY=ON \
      ..
make -j$(nproc)
```

## 测试

仿照 NCCL profile 脚本：不 warmup、只跑一次、关掉结果校验。每次只测一个固定的 message size，这样 CSV 里只对应一次 RCCL 调用。

```bash
LD_LIBRARY_PATH=$(readlink -f ../../rccl/install/lib):$LD_LIBRARY_PATH

# 单次 AllReduce，2GB，只跑一次
NCCL_PRIM_PROFILE=1 \
NCCL_PRIM_PROFILE_FILE=/tmp/rccl_prim_profile.csv \
./build/all_reduce_perf -b 2G -e 2G -g 8 -w 0 -n 1 -c 0

# 小消息测 LL / LL128
NCCL_PRIM_PROFILE=1 \
NCCL_PRIM_PROFILE_FILE=/tmp/rccl_prim_profile.csv \
./build/all_reduce_perf -b 256 -e 256 -g 8 -w 0 -n 1 -c 0

# ReduceScatter
NCCL_PRIM_PROFILE=1 \
NCCL_PRIM_PROFILE_FILE=/tmp/rccl_prim_profile.csv \
./build/reduce_scatter_perf -b 2G -e 2G -g 8 -w 0 -n 1 -c 0

# AllGather
NCCL_PRIM_PROFILE=1 \
NCCL_PRIM_PROFILE_FILE=/tmp/rccl_prim_profile.csv \
./build/all_gather_perf -b 2G -e 2G -g 8 -w 0 -n 1 -c 0

# SendRecv（会触发 P2P profileMode）
NCCL_PRIM_PROFILE=1 \
NCCL_PRIM_PROFILE_FILE=/tmp/rccl_prim_profile.csv \
./build/sendrecv_perf -b 2G -e 2G -g 8 -w 0 -n 1 -c 0

# AlltoAll
NCCL_PRIM_PROFILE=1 \
NCCL_PRIM_PROFILE_FILE=/tmp/rccl_prim_profile.csv \
./build/alltoall_perf -b 2G -e 2G -g 8 -w 0 -n 1 -c 0
```

参数含义：
- `-w 0`：0 次 warmup
- `-n 1`：只跑 1 次
- `-c 0`：不做结果校验
- `-b 2G -e 2G`：固定测 2GB 消息

## 分析

```bash
# 在 projects/rccl 下
python3 tools/rccl_prim_profile_report.py \
  --input /tmp/rccl_prim_profile.csv \
  --outdir /tmp/rccl_prim_report
```

## 日志

```bash
# 同时看 RCCL 日志确认 profiling 开启
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=PROFILE \
NCCL_PRIM_PROFILE=1 \
NCCL_PRIM_PROFILE_FILE=/tmp/rccl_prim_profile.csv \
./build/all_reduce_perf -b 2G -e 2G -g 8 -w 0 -n 1 -c 0
```

如果看到 `PRIMPROF` 开头的日志就说明 profiling 生效了。

## 核心指标（tb_summary.csv）

- `wait_cycles` — primitive 内等待 peer data / recv flag / FIFO credit
- `compute_cycles` — reduce / copy / store 主体工作
- `sync_cycles` — barrier / postPeer / postSend / postRecv / step bookkeeping
- `idle_cycles` — primitive 外部空档
- `dominant_wait_reason` — 主要瓶颈原因
