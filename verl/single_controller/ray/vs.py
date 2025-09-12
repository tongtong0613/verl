# test_execute_all_async_comparison.py （工程化同步接口版）

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
# Executor：对比原始 vs 修改后（都提供同步接口）
# =============================================
class SubmitExecutor:
    def __init__(self, workers):
        self._workers = workers
        self._submission_times = []

    # --------------------------------------------------
    # ✅ 原始版本：串行提交（同步接口）
    # --------------------------------------------------
    def execute_all_async_original(self, method_name: str, *args, **kwargs):
        """原始实现：串行提交"""
        self._submission_times.clear()
        length = len(self._workers)

        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if not (all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values())):
                raise ValueError(f"Argument list lengths must match number of workers ({length})")
        else:
            args = [args] * length
            kwargs = {k: [v] * length for k, v in kwargs.items()}

        result = []
        for i in range(length):
            worker_args = tuple(arg[i] for arg in args)
            worker_kwargs = {k: v[i] for k, v in kwargs.items()}
            ref = self._execute_remote_single_worker(self._workers[i], i, method_name, *worker_args, **worker_kwargs)
            result.append(ref)
        return result

    # --------------------------------------------------
    # ✅ 修改后版本：并行提交（通过同步接口封装异步实现）
    # --------------------------------------------------
    def execute_all_async_after(self, method_name: str, workers: int = 8, *args, **kwargs):
        """同步接口，内部使用 asyncio.run 启动真正的异步实现"""
        return asyncio.run(self._execute_all_async_impl(method_name, workers, *args, **kwargs))

    async def _execute_all_async_impl(self, method_name: str, workers: int, *args, **kwargs):
        """真正的异步并行实现"""
        self._submission_times.clear()
        length = len(self._workers)

        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if not (all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values())):
                raise ValueError(f"Argument list lengths must match number of workers ({length})")
        else:
            args = [args] * length
            kwargs = {k: [v] * length for k, v in kwargs.items()}

        loop = asyncio.get_event_loop()
        futures = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i in range(length):
                worker_args = tuple(arg[i] for arg in args)
                worker_kwargs = {k: v[i] for k, v in kwargs.items()}
                func = partial(self._execute_remote_single_worker, self._workers[i], i, method_name, *worker_args, **worker_kwargs)
                future = loop.run_in_executor(executor, func)
                futures.append(future)

        refs = await asyncio.gather(*futures)
        return refs

    # --------------------------------------------------
    # ✅ 通用的 _execute_remote_single_worker
    # --------------------------------------------------
    def _execute_remote_single_worker(self, worker, worker_id: int, method_name: str, *args, **kwargs):
        submitted_at_str = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        submitted_at = time.perf_counter()
        method = getattr(worker, method_name)
        ref = method.remote(*args, **kwargs)
        remote_done = time.perf_counter()
        print(f"[Main] ⏱️ Worker-{worker_id} .remote() 耗时: {(remote_done - submitted_at) * 1000:.2f} ms")
        self._submission_times.append({
            "worker_id": worker_id,
            "submitted_at": submitted_at_str
        })
        return ref


# =============================================
# ✅ 构造测试数据
# =============================================
def create_test_data(num_workers=8):
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
    delays = []
    for r in results_sorted:
        wid = r['worker_id']
        submit_entry = next(st for st in submission_times if st["worker_id"] == wid)
        submit_sec = time_str_to_seconds(submit_entry['submitted_at'])
        process_sec = time_str_to_seconds(r['processed_at'])
        delay_ms = (process_sec - submit_sec) * 1000
        delays.append(delay_ms)
        print(f"Worker-{wid}: Submitted={submit_entry['submitted_at']}, Started={r['processed_at']}, Delay={delay_ms:.2f} ms")

    max_delay = max(delays)
    min_delay = min(delays)
    print(f"✅ 启动延迟范围: {min_delay:.2f}ms → {max_delay:.2f}ms")
    return delays


# =============================================
# 主测试函数（✅ 全部使用同步接口调用）
# =============================================
def main():
    num_workers = 8
    print(f"🚀 创建 {num_workers} 个 Ray Worker...")
    workers = [Worker.remote(i) for i in range(num_workers)]
    executor = SubmitExecutor(workers)

    input_tensors, delays = create_test_data(num_workers)
    method_name = "compute"
    args = (input_tensors,)
    kwargs = {"delay": delays}

    print("\n" + "=" * 80)
    print("🧪 测试 1: 原始 execute_all_async（串行提交）")
    print("=" * 80)

    start_time = time.perf_counter()
    refs1 = executor.execute_all_async_original(method_name, *args, **kwargs)
    duration1 = time.perf_counter() - start_time
    print(f"[Original] ✅ 所有任务提交完成，耗时: {duration1 * 1000:.2f} ms")

    submission_times1 = sorted(executor._submission_times, key=lambda x: x["worker_id"])
    results1 = ray.get(refs1)
    analyze_start_delays(results1, submission_times1, "原始串行提交")


    print("\n" + "=" * 80)
    print("🧪 测试 2: 修改后 execute_all_async（并行提交）")
    print("=" * 80)

    for thread_count in [1, 2, 4, 8]:
        print(f"\n🔧 使用 {thread_count} 个线程进行并行提交...")

        start_time = time.perf_counter()
        # ✅ 同步调用！用户无感知 async
        refs2 = executor.execute_all_async_after(method_name, thread_count, *args, **kwargs)
        duration2 = time.perf_counter() - start_time
        print(f"[Modified] ✅ 所有任务提交完成，耗时: {duration2 * 1000:.2f} ms")

        submission_times2 = sorted(executor._submission_times, key=lambda x: x["worker_id"])
        results2 = ray.get(refs2)
        analyze_start_delays(results2, submission_times2, "修改后并行提交")


        # ============================
        # 📊 最终对比
        # ============================
        print("\n" + "=" * 80)
        print("📊 提交耗时对比")
        print("=" * 80)
        print(f"原始串行提交耗时: {duration1 * 1000:.2f} ms")
        print(f"修改后并行提交耗时: {duration2 * 1000:.2f} ms")
        if duration2 > 0:
            print(f"✅ 加速比: {duration1 / duration2:.2f}x")


# =============================================
# 入口点
# =============================================
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()

    try:
        main()  # ✅ 全部同步调用
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断。")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
    finally:
        print("\n🔄 正在关闭 Ray...")
        ray.shutdown()
        print("✅ Ray 已关闭。")