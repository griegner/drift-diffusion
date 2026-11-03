"""Bounded spawned fits, including their chain processes (Windows Job Objects)."""

import ctypes
import os
import signal
import time
import traceback
from multiprocessing import get_context


class ProcessTree:
    """Own all descendants of a fit; closing the Windows job kills stragglers."""

    def __init__(self, process):
        self.process = process
        self.handle = None
        if os.name != "nt":
            return
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_int64),
                ("job_time", ctypes.c_int64),
                ("flags", w.DWORD),
                ("min_ws", ctypes.c_size_t),
                ("max_ws", ctypes.c_size_t),
                ("active", w.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", w.DWORD),
                ("scheduling", w.DWORD),
            ]

        class Extended(ctypes.Structure):
            _fields_ = [
                ("basic", Basic),
                ("io", ctypes.c_uint64 * 6),
                ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t),
                ("peak_process", ctypes.c_size_t),
                ("peak_job", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        kernel.QueryInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p]
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            kernel.CloseHandle(handle)
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.AssignProcessToJobObject(handle, int(process.sentinel)):
            kernel.CloseHandle(handle)
            raise ctypes.WinError(ctypes.get_last_error())
        self.kernel, self.handle = kernel, handle

    def close(self):
        if self.handle:
            # Basic accounting: four time counters followed by four DWORDs;
            # active process count is DWORD index 10 (byte offset 40).
            accounting = (ctypes.c_uint64 * 6)()
            try:
                if not self.kernel.TerminateJobObject(self.handle, 1):
                    raise ctypes.WinError(ctypes.get_last_error())
                deadline = time.monotonic() + 10
                while True:
                    if not self.kernel.QueryInformationJobObject(
                        self.handle, 1, accounting, ctypes.sizeof(accounting), None
                    ):
                        raise ctypes.WinError(ctypes.get_last_error())
                    active = ctypes.cast(accounting, ctypes.POINTER(ctypes.c_uint32))[10]
                    if active == 0:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Fit's process tree did not terminate; stopping supervisor")
                    time.sleep(0.02)
            finally:
                self.kernel.CloseHandle(self.handle)
                self.handle = None
        elif os.name != "nt":
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.process.close()


def _worker(gate, pipe, function, args):
    if os.name != "nt":
        os.setsid()
    gate.wait()  # parent assigns job before any sampling children can start
    try:
        pipe.send(("completed", function(*args)))
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
    finally:
        pipe.close()


def bounded_fits(function, tasks, workers, timeout):
    """Yield (task key, outcome, result/error, wall seconds); refill freed slots."""
    context = get_context("spawn")
    pending = iter(tasks)
    active = []
    exhausted = False
    try:
        while active or not exhausted:
            while len(active) < workers and not exhausted:
                try:
                    key, args = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                receive, send = context.Pipe(duplex=False)
                gate = context.Event()
                process = context.Process(target=_worker, args=(gate, send, function, args))
                began = time.monotonic()
                process.start()
                send.close()
                try:
                    tree = ProcessTree(process)
                except BaseException:
                    process.kill()
                    process.join()
                    receive.close()
                    raise
                active.append((key, process, tree, receive, began))
                gate.set()
            for entry in active[:]:
                key, process, tree, pipe, began = entry
                elapsed = time.monotonic() - began
                outcome = None
                if pipe.poll():
                    try:
                        outcome, result = pipe.recv()
                    except EOFError:
                        outcome, result = "error", "Worker exited without a result"
                elif not process.is_alive():
                    outcome, result = "error", f"Worker exit code {process.exitcode}"
                elif elapsed >= timeout:
                    outcome, result = "timeout", f"Exceeded {timeout:g} seconds"
                if outcome is not None:
                    tree.close()
                    pipe.close()
                    active.remove(entry)
                    yield key, outcome, result, time.monotonic() - began
            if active:
                time.sleep(0.1)
    finally:
        for _, _, tree, pipe, _ in active:
            tree.close()
            pipe.close()
