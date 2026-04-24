#!/usr/bin/env python3
"""实时监控脚本：
- 每隔 10 秒检查 `mywebthinker_macro_data_create.py` 是否仍在运行；
- 若脚本正常结束（exit code 0），则认为任务已完成并退出监控；
- 若脚本异常退出（exit code 非 0），等待 10 秒后自动重启，依赖其断点续跑机制跳过已处理样例。

使用方法：
    python scripts/monitor_mywebthinker.py

注：如需在后台长期运行，可结合 `nohup`/`tmux`/`screen` 等工具。
"""

import subprocess
import time
import os
import sys
from datetime import datetime

# 路径配置，根据实际项目结构进行调整
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_SCRIPT = os.path.join(SCRIPT_DIR, "mywebthinker_macro_data_create.py")
PYTHON_EXE = sys.executable  # 当前 Python 解释器

CHECK_INTERVAL = 60  # 秒
RESTART_DELAY = 10    # 进程异常退出后的重启等待时间


def timestamp() -> str:
    """返回当前时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    """主循环，负责启动并监控子进程。"""
    while True:
        print(f"[{timestamp()}] 启动 {TARGET_SCRIPT}……")

        # 启动子进程
        process = subprocess.Popen([PYTHON_EXE, TARGET_SCRIPT])
        print(f"[{timestamp()}] 进程 PID={process.pid} 已启动。")

        # 监控循环：每 CHECK_INTERVAL 秒轮询一次进程状态
        while True:
            time.sleep(CHECK_INTERVAL)
            ret = process.poll()
            if ret is None:
                # 仍在运行
                print(f"[{timestamp()}] 进程 PID={process.pid} 仍在运行……")
                continue

            # 进程已结束
            if ret == 0:
                print(f"[{timestamp()}] 进程正常结束 (exit code 0)，监控脚本退出。")
                return  # 正常完成，无需重启
            else:
                print(
                    f"[{timestamp()}] 检测到进程异常退出 (exit code {ret})，"
                    f"{RESTART_DELAY}s 后自动重启。"
                )
                break  # 跳出监控循环，等待重启

        # 等待一段时间再重启
        time.sleep(RESTART_DELAY)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n[{timestamp()}] 收到 Ctrl+C，监控脚本退出。") 