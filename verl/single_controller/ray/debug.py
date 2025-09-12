# test_ray_submit_fixed.py

import ray
import torch
import time
import asyncio
import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tensordict import TensorDict

# ====================================================
# ✅ 启动 Ray
# ====================================================
ray.init(num_cpus=8, log_to_driver=True)

# =============================================
# Worker 类
# =============================================
@ray.remote
class Worker:
    def __init__(self, worker_id):
        self.worker_id = worker_id

    def compute(self, data: TensorDict, delay: float = 1.0):
        ts_start = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[Worker-{self.worker_id}] {ts_start} | Starting compute, batch_size={data.batch_size}")
        time.sleep(delay)
        return {
            "worker_id": self.worker_id,
            "batch_size": int(data.batch_size[0]),
            "processed_at": ts_start,
        }

# =============================================
# Executor：支持三种提交方式（修复版）
# =============================================
class SubmitExecutor:
    def __init__(self, workers):
        self._workers = workers

    # --------------------------------------------------
    # 方式1：同步串行提交（baseline）
    # --------------------------------------------------
    def execute_sync(self, method_name, args_list, kwargs_list):
        refs = []
        submission_times = []

        print(f"[SyncSubmit] Start submitting {len(self._workers)} tasks...")
        start_total = time.perf_counter()

        for i in range(len(self._workers)):
            current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f"[SyncSubmit-{i}] {current_time} | Submitting to worker-{i}")
            method = getattr(self._workers[i], method_name)
            ref = method.remote(*args_list[i], **kwargs_list[i])
            refs.append(ref)
            submission_times.append({
                'worker_id': i,
                'submitted_at': current_time,
                'expected_delay': kwargs_list[i].get('delay', 1.0)
            })

        end_total = time.perf_counter()
        duration_ms = (end_total - start_total) * 1000
        print(f"[SyncSubmit] ✅ All tasks submitted in {duration_ms:.2f} ms")

        return refs, submission_times

    # --------------------------------------------------
    # 方式2：旧版“异步提交”（整个 for 循环放线程池）
    # --------------------------------------------------
    def _submit_sync_helper(self, method_name, args_list, kwargs_list):
        refs = []
        submission_times = []
        start_local = time.perf_counter()

        for i in range(len(self._workers)):
            current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f"[AsyncOld-{i}] {current_time} | Submitting to worker-{i}")
            method = getattr(self._workers[i], method_name)
            ref = method.remote(*args_list[i], **kwargs_list[i])
            refs.append(ref)
            submission_times.append({
                'worker_id': i,
                'submitted_at': current_time,
                'expected_delay': kwargs_list[i].get('delay', 1.0)
            })

        end_local = time.perf_counter()
        print(f"[AsyncOld] Thread pool submission took: {(end_local - start_local)*1000:.2f} ms")
        return refs, submission_times

    async def execute_async_old(self, method_name, args_list, kwargs_list):
        loop = asyncio.get_event_loop()
        start_total = time.perf_counter()

        with ThreadPoolExecutor(max_workers=8) as executor:
            refs, submission_times = await loop.run_in_executor(
                executor, self._submit_sync_helper, method_name, args_list, kwargs_list
            )

        end_total = time.perf_counter()
        duration_ms = (end_total - start_total) * 1000
        print(f"[AsyncOld] ✅ All tasks submitted in {duration_ms:.2f} ms")

        return refs, submission_times

    # --------------------------------------------------
    # 方式3：真正并行提交（每个 .remote() 独立线程）
    # --------------------------------------------------
    async def execute_parallel_submit(self, method_name, max_workers, args_list, kwargs_list):
        loop = asyncio.get_event_loop()
        start_total = time.perf_counter()

        print(f"[ParallelSubmit] Start parallel submitting {len(self._workers)} tasks...")

        submission_times = []  # 用于收集每个任务的真实提交时间
        futures = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(len(self._workers)):
                method = getattr(self._workers[i], method_name)

                # ✅ 封装 remote 调用 + 时间记录 在同一个函数中
                def make_remote_call(index, method, args, kwargs):
                    # 🔥 关键：时间记录和 .remote() 发生在同一线程、同一时刻
                    submitted_at = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    ref = method.remote(*args, **kwargs)
                    return {"worker_id": index, "ref": ref, "submitted_at": submitted_at}

                # 使用 partial 绑定参数
                bound_call = partial(make_remote_call, i, method, args_list[i], kwargs_list[i])
                future = loop.run_in_executor(executor, bound_call)
                futures.append(future)

            # 等待所有提交完成，并获取结果
            results = await asyncio.gather(*futures)

        # 分离 refs 和 submission_times
        refs = [r["ref"] for r in results]
        submission_times = [
            {"worker_id": r["worker_id"], "submitted_at": r["submitted_at"]}
            for r in results
        ]

        end_total = time.perf_counter()
        duration_ms = (end_total - start_total) * 1000
        print(f"[ParallelSubmit] ✅ All tasks submitted in {duration_ms:.2f} ms")

        return refs, submission_times


# =============================================
# ✅ 构造测试数据
# =============================================
def create_test_data(num_workers=4):
    batch_size = 1024
    seq_len = 1056
    input_tensors = []
    for i in range(num_workers):
        td = TensorDict(
            {
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.int64),
                "input_ids": torch.randint(0, 32000, (batch_size, seq_len), dtype=torch.int64),
                "position_ids": torch.randint(0, seq_len, (batch_size, seq_len), dtype=torch.int64),
            },
            batch_size=[batch_size]
        )
        input_tensors.append(td)
    delays = [1.0] * num_workers
    return input_tensors, delays


# =============================================
# 工具函数：时间字符串转秒
# =============================================
def time_str_to_seconds(t: str) -> float:
    h, m, s = t.split(':')
    s, ms = s.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


# =============================================
# 分析函数：计算启动延迟
# =============================================
def analyze_start_delays(results, submission_times, label=""):
    print(f"\n📈 {label} 启动延迟分析:")
    print("-" * 60)
    results_sorted = sorted(results, key=lambda x: x['worker_id'])
    submission_times_sorted = sorted(submission_times, key=lambda x: x['worker_id'])

    delays = []
    for r, s in zip(results_sorted, submission_times_sorted):
        wid = r['worker_id']
        submit_sec = time_str_to_seconds(s['submitted_at'])
        process_sec = time_str_to_seconds(r['processed_at'])
        delay_ms = (process_sec - submit_sec) * 1000
        delays.append(delay_ms)
        print(f"Worker-{wid}: Submitted={s['submitted_at']}, Started={r['processed_at']}, Delay={delay_ms:.2f} ms")

    max_delay = max(delays)
    min_delay = min(delays)
    print(f"✅ 启动延迟范围: {min_delay:.2f}ms → {max_delay:.2f}ms")
    return delays


# =============================================
# 主测试函数
# =============================================
async def main():
    num_workers = 8
    print(f"🚀 创建 {num_workers} 个 Ray Worker...")
    workers = [Worker.remote(i) for i in range(num_workers)]
    executor = SubmitExecutor(workers)

    input_tensors, delays = create_test_data(num_workers)

    # 构造 args_list 和 kwargs_list
    args_list = [(input_tensors[i],) for i in range(num_workers)]
    kwargs_list = [{'delay': delays[i]} for i in range(num_workers)]

    print("\n" + "=" * 80)
    print("🧪 测试 1: 同步串行提交")
    print("=" * 80)
    sync_refs, sync_submit_times = executor.execute_sync("compute", args_list, kwargs_list)
    sync_results = ray.get(sync_refs)
    analyze_start_delays(sync_results, sync_submit_times, "同步串行提交")

    print("\n" + "=" * 80)
    print("🧪 测试 2: 旧版异步提交（for-loop 在线程池）")
    print("=" * 80)
    async_old_refs, async_old_submit_times = await executor.execute_async_old("compute", args_list, kwargs_list)
    async_old_results = ray.get(async_old_refs)
    analyze_start_delays(async_old_results, async_old_submit_times, "旧版异步提交")

    print("\n" + "=" * 80)
    print("🧪 测试 3: 真正并行提交（每个 .remote() 独立线程）")
    print("=" * 80)
    for workers in [1,2,4,8]:
        parallel_refs, parallel_submit_times = await executor.execute_parallel_submit("compute", workers, args_list, kwargs_list)
        parallel_results = ray.get(parallel_refs)
        analyze_start_delays(parallel_results, parallel_submit_times, "并行提交")

    # ============================
    # 📊 最终对比（需手动记录日志中的耗时）
    # ============================
    print("\n" + "=" * 80)
    print("📊 提交耗时对比（查看日志中的 ✅ 提交耗时）")
    print("=" * 80)
    print("请查看上方日志中每种方式的 '✅ All tasks submitted in X.XX ms'")
    print("预期：并行提交 << 同步/异步旧版")


# =============================================
# 入口点
# =============================================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(main())
    finally:
        ray.shutdown()