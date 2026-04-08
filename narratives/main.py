import subprocess
import sys
import time
import os
import shutil
import urllib.request


def _launch(script, log_path, old_log=None, cmd=None):
    if old_log:
        try:
            old_log.close()
        except OSError:
            pass
    log  = open(log_path, "a")
    args = cmd if cmd else [sys.executable, script]
    proc = subprocess.Popen(args, stdout=log, stderr=log)
    return proc, log


def _wait_for_chroma(timeout=30):
    url = "http://localhost:8001/api/v2/heartbeat"
    for _ in range(timeout):
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _chroma_cmd():
    chroma_bin = shutil.which("chroma") or os.path.join(os.path.dirname(sys.executable), "chroma")
    return [chroma_bin, "run", "--path", "./db/chroma_vector_db", "--port", "8001"]


def _log_crash(name, proc):
    code = proc.returncode
    print(f"[Orchestrator] WARNING: {name} crashed! Exit code: {code}. Restarting...", flush=True)


def main():
    print("[Orchestrator] Booting pipeline...", flush=True)
    os.makedirs("./db/chroma_vector_db", exist_ok=True)

    print("[Orchestrator] Launching ChromaDB server...", flush=True)
    chroma, chroma_log = _launch(None, "chroma.log", cmd=_chroma_cmd())
    if not _wait_for_chroma():
        print("[Orchestrator] FATAL: ChromaDB server did not start. Check chroma.log.", flush=True)
        sys.exit(1)
    print("[Orchestrator] ChromaDB server ready.", flush=True)

    try:
        import database
        database.init_db()
        database.get_chroma_collection()
        print("[Orchestrator] Databases verified successfully.", flush=True)
    except Exception as e:
        print(f"[Orchestrator] FATAL ERROR during DB init: {e}", flush=True)
        sys.exit(1)

    print("[Orchestrator] Launching Watchdog Daemon...", flush=True)
    watchdog, watchdog_log = _launch("watchdog_daemon.py", "watchdog.log")
    time.sleep(5)

    print("[Orchestrator] Launching Periodic Analysis Scheduler...", flush=True)
    scheduler, scheduler_log = _launch("periodic_analysis.py", "scheduler.log")

    try:
        while True:
            time.sleep(10)

            if chroma.poll() is not None:
                _log_crash("ChromaDB server", chroma)
                chroma, chroma_log = _launch(None, "chroma.log", chroma_log, cmd=_chroma_cmd())
                if not _wait_for_chroma():
                    print("[Orchestrator] FATAL: ChromaDB server failed to recover.", flush=True)
                    sys.exit(1)
                print("[Orchestrator] ChromaDB server recovered.", flush=True)

            if watchdog.poll() is not None:
                _log_crash("Watchdog Daemon", watchdog)
                watchdog, watchdog_log = _launch("watchdog_daemon.py", "watchdog.log", watchdog_log)

            if scheduler.poll() is not None:
                _log_crash("Periodic Analysis", scheduler)
                scheduler, scheduler_log = _launch("periodic_analysis.py", "scheduler.log", scheduler_log)

    except KeyboardInterrupt:
        print("\n[Orchestrator] Shutting down all processes...", flush=True)
        watchdog.terminate()
        scheduler.terminate()
        chroma.terminate()
        watchdog.wait()
        scheduler.wait()
        chroma.wait()
        watchdog_log.close()
        scheduler_log.close()
        chroma_log.close()
        print("[Orchestrator] Offline.", flush=True)


if __name__ == "__main__":
    main()