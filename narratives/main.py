# main.py

import subprocess
import sys
import time
import os

SIGNAL_FILE = "./clusters/.analysis_needed"
LOCK_FILE = "./clusters/.analysis_running"
OUTPUT_FILE = "./clusters/narratives_latest.json"


def _launch(script, log_path, old_log=None, extra_args=None):
    if old_log:
        try:
            old_log.close()
        except OSError:
            pass
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "a")
    cmd = [sys.executable, "-u", script] + (extra_args or [])
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    return proc, log


def _log_crash(name, proc):
    print(
        f"[Orchestrator] WARNING: {name} crashed (exit {proc.returncode}). Restarting...",
        flush=True,
    )


def _wait_for_analysis_complete(timeout=7200):
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(LOCK_FILE):
            print("[Orchestrator] Analysis still running, waiting...", flush=True)
            time.sleep(30)
            continue
        if os.path.exists(OUTPUT_FILE):
            try:
                mtime = os.path.getmtime(OUTPUT_FILE)
                if mtime > start:
                    print("[Orchestrator] Fresh analysis output detected.", flush=True)
                    return True
            except OSError:
                pass
        time.sleep(10)
    print("[Orchestrator] Timeout waiting for analysis.", flush=True)
    return False


def main():
    print("[Orchestrator] Booting pipeline...", flush=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./clusters", exist_ok=True)

    try:
        import database
        database.init_db()
        print("[Orchestrator] Database initialized.", flush=True)
    except Exception as e:
        print(f"[Orchestrator] FATAL: DB init failed: {e}", flush=True)
        sys.exit(1)

    for stale_file in [SIGNAL_FILE, LOCK_FILE]:
        if os.path.exists(stale_file):
            try:
                os.remove(stale_file)
                print(f"[Orchestrator] Removed stale {stale_file}", flush=True)
            except OSError:
                pass

    print("[Orchestrator] Launching Watchdog Daemon...", flush=True)
    watchdog, watchdog_log = _launch("watchdog_daemon.py", "./logs/watchdog.log")
    time.sleep(3)

    print("[Orchestrator] Launching Analysis Scheduler...", flush=True)
    scheduler, scheduler_log = _launch("analysis.py", "./logs/scheduler.log")

    watchdog_done = False
    final_analysis_started = False

    try:
        while True:
            time.sleep(10)

            if not watchdog_done and watchdog.poll() is not None:
                if watchdog.returncode == 0:
                    print("[Orchestrator] Watchdog finished successfully.", flush=True)
                    watchdog_done = True

                    if os.path.exists(LOCK_FILE):
                        print("[Orchestrator] Analysis currently running, waiting for completion...", flush=True)
                        _wait_for_analysis_complete(timeout=3600)

                    try:
                        scheduler.terminate()
                        scheduler.wait(timeout=30)
                    except Exception:
                        try:
                            scheduler.kill()
                            scheduler.wait(timeout=10)
                        except Exception:
                            pass

                    print("[Orchestrator] Signaling final analysis...", flush=True)
                    open(SIGNAL_FILE, "w").close()

                    print("[Orchestrator] Launching final analysis...", flush=True)
                    scheduler, scheduler_log = _launch(
                        "analysis.py", "./logs/scheduler.log", scheduler_log,
                        extra_args=["--once"]
                    )
                    final_analysis_started = True

                else:
                    _log_crash("Watchdog Daemon", watchdog)
                    watchdog, watchdog_log = _launch(
                        "watchdog_daemon.py", "./logs/watchdog.log", watchdog_log
                    )

            if not watchdog_done and scheduler.poll() is not None:
                if scheduler.returncode != 0:
                    _log_crash("Periodic Analysis", scheduler)
                scheduler, scheduler_log = _launch(
                    "analysis.py", "./logs/scheduler.log", scheduler_log
                )

            if watchdog_done and scheduler.poll() is not None:
                if scheduler.returncode == 0:
                    print("[Orchestrator] Final analysis complete.", flush=True)
                else:
                    print(f"[Orchestrator] Final analysis failed (exit {scheduler.returncode}).", flush=True)
                    if final_analysis_started:
                        print("[Orchestrator] Retrying final analysis...", flush=True)
                        scheduler, scheduler_log = _launch(
                            "analysis.py", "./logs/scheduler.log", scheduler_log,
                            extra_args=["--once"]
                        )
                        continue

                if os.path.exists(OUTPUT_FILE):
                    print(f"[Orchestrator] Output file exists: {OUTPUT_FILE}", flush=True)
                else:
                    print(f"[Orchestrator] WARNING: Output file not found: {OUTPUT_FILE}", flush=True)

                break

    except KeyboardInterrupt:
        print("\n[Orchestrator] Shutting down...", flush=True)
        for proc in (watchdog, scheduler):
            try:
                proc.terminate()
            except OSError:
                pass
        for proc in (watchdog, scheduler):
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        for log in (watchdog_log, scheduler_log):
            try:
                log.close()
            except OSError:
                pass
        print("[Orchestrator] Offline.", flush=True)

    finally:
        for log in (watchdog_log, scheduler_log):
            try:
                log.close()
            except OSError:
                pass

    if os.path.exists(OUTPUT_FILE):
        print("[Orchestrator] Pipeline completed successfully.", flush=True)
        sys.exit(0)
    else:
        print("[Orchestrator] Pipeline completed but no output generated.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()