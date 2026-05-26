# RCCL Primitive Profiler

## 编译

```bash
# 在 projects/rccl 下
./install.sh -f --prefix $PWD/install

# 在 projects/rccl-tests 下
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=$(readlink -f ../rccl/install) \
      -DBUILD_LOCAL_GPU_TARGET_ONLY=ON \
      ..
make -j$(nproc)
```

## 测试

仿照 NCCL profile 脚本：不 warmup、只跑一次、关掉结果校验。每次只测一个固定的 message size，这样 CSV 里只对应一次 RCCL 调用。

```bash
cd rccl-tests

rm /tmp/rccl_prim_profile.csv

NCCL_MAX_NCHANNELS=128 \
NCCL_MIN_NCHANNELS=128 \
NCCL_PRIM_PROFILE=1 \
NCCL_PRIM_PROFILE_FILE=/tmp/rccl_prim_profile.csv \
LD_PRELOAD=$(readlink -f ../rccl/install/lib/librccl.so) \
./build/all_reduce_perf -b 1G -e 1G -g 8 -w 0 -n 1 -c 0

LD_LIBRARY_PATH=$(readlink -f ../rccl/install/lib):$LD_LIBRARY_PATH
# AllReduce
./build/all_reduce_perf
# ReduceScatter
./build/reduce_scatter_perf
# AllGather
./build/all_gather_perf
# AlltoAll
./build/alltoall_perf
```

参数含义：
- `-w 0`：0 次 warmup
- `-n 1`：只跑 1 次
- `-c 0`：不做结果校验
- `-b 2G -e 2G`：固定测 2GB 消息

## 分析

```bash
# 在 projects/rccl 下
cd rccl
rm -r /tmp/rccl_prim_report

python3 tools/rccl_prim_profile_report.py \
  --input /tmp/rccl_prim_profile.csv \
  --outdir /tmp/rccl_prim_report

mkdir ~/jinyao/rccl-results/allreduce-8gpus-128channels
cp -r /tmp/rccl_prim_report ~/jinyao/rccl-results/allreduce-8gpus-128channels
```

