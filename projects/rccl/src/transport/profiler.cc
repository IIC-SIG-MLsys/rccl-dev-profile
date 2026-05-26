/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "transport.h"
#include "proxy.h"
#include "profiler.h"
#include "device.h"
#include "debug.h"
#include "param.h"
#include <cstdio>
#include <mutex>

static const char* ncclPrimProfileName[ncclPrimN] = {
  "send",
  "sendFromOutput",
  "directSend",
  "directSendFromOutput",
  "recv",
  "directRecv",
  "directRecvCopy",
  "copySend",
  "directCopySend",
  "recvSend",
  "recvCopySend",
  "directRecvCopyDirectSend",
  "directRecvDirectSend",
  "recvDirectSend",
  "directRecvSend",
  "recvCopyDirectSend",
  "recvReduceCopy",
  "directRecvReduceCopy",
  "recvReduceSend",
  "directRecvReduceSend",
  "recvReduceDirectSend",
  "directRecvReduceDirectSend",
  "recvReduceCopySend",
  "recvReduceCopyDirectSend",
  "directRecvReduceCopyDirectSend",
};

static const char* ncclTbStageName[ncclTbStageN] = {
  "wait",
  "compute",
  "sync",
};

static std::mutex ncclPrimProfileFileMutex;
static FILE* ncclPrimProfileFile = nullptr;
static bool ncclPrimProfileFileInit = false;

static FILE* ncclPrimProfileGetFile() {
  std::lock_guard<std::mutex> lock(ncclPrimProfileFileMutex);
  if (!ncclPrimProfileFileInit) {
    ncclPrimProfileFileInit = true;
    const char* filePath = ncclGetEnv("NCCL_PRIM_PROFILE_FILE");
    if (filePath != nullptr && filePath[0] != '\0') {
      ncclPrimProfileFile = fopen(filePath, "a");
      if (ncclPrimProfileFile == nullptr) {
        WARN("Could not open NCCL_PRIM_PROFILE_FILE=%s", filePath);
      } else {
        fseek(ncclPrimProfileFile, 0, SEEK_END);
        long fileSize = ftell(ncclPrimProfileFile);
        if (fileSize == 0) {
          fprintf(ncclPrimProfileFile, "type,op_count,channel,work,tb_cycles,prim_cycles_total,wait_cycles,compute_cycles,sync_cycles,prim,cycles,calls,pct_tb,pct_prim_sum,start_clk,stop_clk,trace_group,trace_seq,trace_start,trace_stop,trace_dur,trace_start_off,trace_stop_off,trace_dropped\n");
        }
        fflush(ncclPrimProfileFile);
      }
    }
  }
  return ncclPrimProfileFile;
}

static inline void ncclProfilerLogPrimSummary(
    const struct ncclProxyArgs* args,
    const struct ncclProxySubArgs* sub,
    const struct ncclDevProfilerStartRecord* startRec,
    const struct ncclDevProfilerRecord* stopRec
  ) {
  if (!ncclPrimProfileEnabled()) return;
  static int callCount = 0;
  if (callCount < 3) {
    fprintf(stderr, "[RCCL-PROFILE-DEBUG] ncclProfilerLogPrimSummary() called, opCount=%llu\n", (unsigned long long)args->opCount);
    fflush(stderr);
    callCount++;
  }
  if (stopRec->counter < sub->base) {
    fprintf(stderr, "[RCCL-PROFILE-DEBUG] LogPrim counter=%llu < base=%llu, skipping\n", (unsigned long long)stopRec->counter, (unsigned long long)sub->base);
    fflush(stderr);
    return;
  }

  uint64_t startClk = stopRec->tbStart;
  uint64_t stopClk = stopRec->tbStop;
  if (stopClk <= startClk) {
    startClk = startRec->timestamp;
    stopClk = stopRec->timestamp;
  }
  if (stopClk <= startClk) return;

  uint64_t tbCycles = stopClk - startClk;
  uint64_t primCyclesTotal = 0;
  for (int p = 0; p < ncclPrimN; p++) primCyclesTotal += stopRec->primCycles[p];
  uint64_t waitCycles = stopRec->stageCycles[ncclTbStageWait];
  uint64_t syncCycles = stopRec->stageCycles[ncclTbStageSync];
  uint64_t computeCycles = stopRec->stageCycles[ncclTbStageCompute];
  if (primCyclesTotal >= waitCycles + syncCycles) {
    computeCycles = primCyclesTotal - waitCycles - syncCycles;
  }

  INFO(NCCL_PROFILE,
       "PRIMPROF op_count %llu channel %d work %llu tb_cycles %llu start_clk %llu stop_clk %llu prim_cycles_total %llu wait_cycles %llu compute_cycles %llu sync_cycles %llu",
       (unsigned long long)args->opCount,
       sub->channelId,
       (unsigned long long)sub->base,
       (unsigned long long)tbCycles,
       (unsigned long long)startClk,
       (unsigned long long)stopClk,
       (unsigned long long)primCyclesTotal,
       (unsigned long long)waitCycles,
       (unsigned long long)computeCycles,
       (unsigned long long)syncCycles);

  FILE* f = ncclPrimProfileGetFile();

  for (int p = 0; p < ncclPrimN; p++) {
    if (stopRec->primCycles[p] == 0) continue;
    double pctTb = 100.0 * (double)stopRec->primCycles[p] / (double)tbCycles;
    double pctPrim = (primCyclesTotal == 0) ? 0.0 : (100.0 * (double)stopRec->primCycles[p] / (double)primCyclesTotal);
    INFO(NCCL_PROFILE,
         "PRIMPROF op_count %llu channel %d work %llu prim %s cycles %llu calls %u pct_tb %.2f pct_prim_sum %.2f",
         (unsigned long long)args->opCount,
         sub->channelId,
         (unsigned long long)sub->base,
         ncclPrimProfileName[p],
         (unsigned long long)stopRec->primCycles[p],
         stopRec->primCalls[p],
         pctTb,
         pctPrim);

  }
  for (int s = 0; s < ncclTbStageN; s++) {
    uint64_t stageCycles = (s == ncclTbStageCompute) ? computeCycles : stopRec->stageCycles[s];
    if (stageCycles == 0) continue;
    INFO(NCCL_PROFILE,
         "PRIMPROF op_count %llu channel %d work %llu stage %s cycles %llu pct_tb %.2f pct_prim_sum %.2f",
         (unsigned long long)args->opCount,
         sub->channelId,
         (unsigned long long)sub->base,
         ncclTbStageName[s],
         (unsigned long long)stageCycles,
         100.0 * (double)stageCycles / (double)tbCycles,
         (primCyclesTotal == 0) ? 0.0 : (100.0 * (double)stageCycles / (double)primCyclesTotal));
  }
  if (f != nullptr) {
    std::lock_guard<std::mutex> lock(ncclPrimProfileFileMutex);
    fprintf(f, "tb,%llu,%d,%llu,%llu,%llu,%llu,%llu,%llu,,0,0,0.00,0.00,%llu,%llu,,,,,,,,%u\n",
            (unsigned long long)args->opCount,
            sub->channelId,
            (unsigned long long)sub->base,
            (unsigned long long)tbCycles,
            (unsigned long long)primCyclesTotal,
            (unsigned long long)waitCycles,
            (unsigned long long)computeCycles,
            (unsigned long long)syncCycles,
            (unsigned long long)startClk,
            (unsigned long long)stopClk,
            stopRec->primTraceDropped);

    for (int p = 0; p < ncclPrimN; p++) {
      if (stopRec->primCycles[p] == 0) continue;
      double pctTb = 100.0 * (double)stopRec->primCycles[p] / (double)tbCycles;
      double pctPrim = (primCyclesTotal == 0) ? 0.0 : (100.0 * (double)stopRec->primCycles[p] / (double)primCyclesTotal);
      fprintf(f, "prim,%llu,%d,%llu,%llu,%llu,%llu,%llu,%llu,%s,%llu,%u,%.6f,%.6f,%llu,%llu,,,,,,,,%u\n",
              (unsigned long long)args->opCount,
              sub->channelId,
              (unsigned long long)sub->base,
              (unsigned long long)tbCycles,
              (unsigned long long)primCyclesTotal,
              (unsigned long long)waitCycles,
              (unsigned long long)computeCycles,
              (unsigned long long)syncCycles,
              ncclPrimProfileName[p],
              (unsigned long long)stopRec->primCycles[p],
              stopRec->primCalls[p],
              pctTb,
              pctPrim,
              (unsigned long long)startClk,
              (unsigned long long)stopClk,
              stopRec->primTraceDropped);
    }

    for (uint32_t t = 0; t < stopRec->primTraceCount && t < NCCL_PRIM_TRACE_MAX_PER_WORK; t++) {
      struct ncclDevPrimTraceEvent const* ev = &stopRec->primTrace[t];
      if (ev->stop <= ev->start) continue;
      if (ev->kind >= ncclPrimN) continue;
      uint64_t dur = ev->stop - ev->start;
      int64_t offStart = (int64_t)ev->start - (int64_t)startClk;
      int64_t offStop = (int64_t)ev->stop - (int64_t)startClk;
      double pctTb = 100.0 * (double)dur / (double)tbCycles;
      fprintf(f, "trace,%llu,%d,%llu,%llu,%llu,%llu,%llu,%llu,%s,%llu,1,%.6f,0.000000,%llu,%llu,%u,%u,%llu,%llu,%llu,%lld,%lld,%u\n",
              (unsigned long long)args->opCount,
              sub->channelId,
              (unsigned long long)sub->base,
              (unsigned long long)tbCycles,
              (unsigned long long)primCyclesTotal,
              (unsigned long long)waitCycles,
              (unsigned long long)computeCycles,
              (unsigned long long)syncCycles,
              ncclPrimProfileName[ev->kind],
              (unsigned long long)dur,
              pctTb,
              (unsigned long long)startClk,
              (unsigned long long)stopClk,
              (unsigned int)ev->group,
              ev->seq,
              (unsigned long long)ev->start,
              (unsigned long long)ev->stop,
              (unsigned long long)dur,
              (long long)offStart,
              (long long)offStop,
              stopRec->primTraceDropped);
    }
    fflush(f);
  }
}

static ncclResult_t profilerProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  connection->proxyAppendPtr = &connection->proxyAppend;
  connection->shared = 0;
  return ncclSuccess;
}

static ncclResult_t profilerProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  static int callCount = 0;
  if (callCount < 10) {
    fprintf(stderr, "[RCCL-PROFILE-DEBUG] profilerProxyProgress() state=%d nsubs=%d stop=%d\n",
            (int)args->state, args->nsubs, proxyState->progressState.stop);
    fflush(stderr);
    callCount++;
  }
  if (args->state == ncclProxyOpReady) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      sub->base = sub->workCounter;
      sub->posted = sub->transmitted = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    int stopping = proxyState->progressState.stop;
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      struct ncclDevProfilerStart* workStarted = (struct ncclDevProfilerStart *)sub->sendbuff;
      struct ncclDevProfiler* workCompleted = (struct ncclDevProfiler *)sub->recvbuff;
      int idx = sub->base % MAX_PROFILER_EVENTS_PER_CHANNEL;
      static int debugCount = 0;
      if (debugCount < 10) {
        fprintf(stderr, "[RCCL-PROFILE-DEBUG] ch=%d base=%llu posted=%d transmitted=%d started_ctr=%llu completed_ctr=%llu\n",
                sub->channelId, (unsigned long long)sub->base, sub->posted, sub->transmitted,
                (unsigned long long)workStarted[sub->channelId].data[idx].counter,
                (unsigned long long)workCompleted[sub->channelId].data[idx].counter);
        fflush(stderr);
        debugCount++;
      }
      if (sub->posted < sub->nsteps) {
        if (sub->base <= workStarted[sub->channelId].data[idx].counter) {
          ncclProfilerStartKernelChEvent(args, s, workStarted[sub->channelId].data[idx].timestamp);
          sub->posted = sub->nsteps;
          continue;
        }
        if (!stopping) continue;
        sub->posted = sub->transmitted = sub->nsteps;
        args->done++;
        continue;
      }
      if (sub->transmitted < sub->nsteps) {
        if (sub->base <= workCompleted[sub->channelId].data[idx].counter) {
          ncclProfilerLogPrimSummary(args, sub, &workStarted[sub->channelId].data[idx], &workCompleted[sub->channelId].data[idx]);
          ncclProfilerStopKernelChEvent(args, s, workCompleted[sub->channelId].data[idx].timestamp);
          sub->transmitted = sub->nsteps;
          args->done++;
        } else if (stopping) {
          sub->transmitted = sub->nsteps;
          args->done++;
        }
      }
    }
    if (args->done == args->nsubs) args->state = ncclProxyOpNone;
  }
  return ncclSuccess;
}

struct ncclTransport profilerTransport = {
  "Prof",
  NULL,
  { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
  { NULL, NULL, NULL, NULL, NULL, profilerProxyConnect, NULL, profilerProxyProgress, NULL, NULL }
};
