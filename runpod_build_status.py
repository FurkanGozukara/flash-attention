#!/usr/bin/env python3
"""Show Linux/RunPod build progress for a Ninja-backed CUDA extension build.

Default paths match a RunPod flash-attn checkout:
  /workspace/flash/build/temp.linux-x86_64-cpython-311
  /workspace/flash/build/lib.linux-x86_64-cpython-311

No third-party dependencies are required.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path("/workspace/flash")
DEFAULT_TEMP_DIR = DEFAULT_ROOT / "build" / "temp.linux-x86_64-cpython-311"
DEFAULT_LIB_DIR = DEFAULT_ROOT / "build" / "lib.linux-x86_64-cpython-311"
BUILD_NAMES = {
    "python",
    "python3",
    "ninja",
    "nvcc",
    "cicc",
    "ptxas",
    "cudafe++",
    "gcc",
    "g++",
    "cc1",
    "cc1plus",
    "clang",
    "clang++",
    "ld",
    "ld.lld",
    "collect2",
}


@dataclass
class Proc:
    pid: int
    ppid: int
    name: str
    cmd: str
    cwd: str
    cpu_seconds: float
    rss_mb: float
    start_ts: float


def fmt_dt(ts: float | None) -> str:
    if not ts:
        return "unknown"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S %Z").strip()


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def proc_boot_time() -> float:
    for line in read_text(Path("/proc/stat")).splitlines():
        if line.startswith("btime "):
            return float(line.split()[1])
    return time.time()


def iter_procs() -> list[Proc]:
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []

    clk_tck = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", "SC_CLK_TCK"))
    page_size = os.sysconf(os.sysconf_names.get("SC_PAGE_SIZE", "SC_PAGE_SIZE"))
    boot = proc_boot_time()
    out: list[Proc] = []

    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat_raw = (entry / "stat").read_text(errors="replace")
            rparen = stat_raw.rfind(")")
            if rparen == -1:
                continue
            comm_start = stat_raw.find("(")
            name = stat_raw[comm_start + 1 : rparen]
            fields = stat_raw[rparen + 2 :].split()
            ppid = int(fields[1])
            utime = int(fields[11])
            stime = int(fields[12])
            start_ticks = int(fields[19])

            cmd_raw = (entry / "cmdline").read_bytes()
            cmd = cmd_raw.replace(b"\0", b" ").decode(errors="replace").strip()
            if not cmd:
                cmd = name

            try:
                cwd = os.readlink(entry / "cwd")
            except OSError:
                cwd = ""

            rss_mb = 0.0
            statm = read_text(entry / "statm").split()
            if len(statm) >= 2:
                rss_mb = int(statm[1]) * page_size / (1024 * 1024)

            out.append(
                Proc(
                    pid=pid,
                    ppid=ppid,
                    name=name,
                    cmd=cmd,
                    cwd=cwd,
                    cpu_seconds=(utime + stime) / clk_tck,
                    rss_mb=rss_mb,
                    start_ts=boot + start_ticks / clk_tck,
                )
            )
        except (OSError, ValueError, IndexError, ProcessLookupError):
            continue
    return out


def path_inside(path_text: str, roots: Iterable[Path]) -> bool:
    if not path_text:
        return False
    for root in roots:
        root_text = str(root)
        if path_text == root_text or path_text.startswith(root_text.rstrip("/") + "/"):
            return True
    return False


def select_build_tree(procs: list[Proc], root: Path, temp_dir: Path, lib_dir: Path) -> list[Proc]:
    roots = [root, temp_dir, lib_dir]
    self_pid = os.getpid()
    proc_by_pid = {p.pid: p for p in procs}
    children: dict[int, list[int]] = defaultdict(list)
    for p in procs:
        if p.pid == self_pid:
            continue
        children[p.ppid].append(p.pid)

    root_pids: set[int] = set()
    for p in procs:
        if p.pid == self_pid:
            continue
        name = p.name.lower()
        cmd = p.cmd
        if path_inside(p.cwd, roots) or any(str(r) in cmd for r in roots):
            if name in BUILD_NAMES or "setup.py" in cmd or "bdist_wheel" in cmd:
                root_pids.add(p.pid)
        if name == "ninja" and ("ninja" in cmd or path_inside(p.cwd, roots)):
            root_pids.add(p.pid)

    selected: set[int] = set(root_pids)
    stack = list(root_pids)
    while stack:
        pid = stack.pop()
        for child_pid in children.get(pid, []):
            if child_pid not in selected:
                selected.add(child_pid)
                stack.append(child_pid)

    if not selected:
        # Fallback for minimal containers where cwd/cmdline is hidden.
        selected = {p.pid for p in procs if p.pid != self_pid and p.name.lower() in BUILD_NAMES}

    return [proc_by_pid[pid] for pid in sorted(selected) if pid in proc_by_pid]


def find_build_file(temp_dir: Path, filename: str) -> Path | None:
    direct = temp_dir / filename
    if direct.exists():
        return direct
    for sub in ("Release", "Debug"):
        candidate = temp_dir / sub / filename
        if candidate.exists():
            return candidate
    if temp_dir.exists():
        try:
            return next(temp_dir.rglob(filename))
        except (OSError, StopIteration):
            return None
    return None


def parse_ninja_log(log_path: Path | None) -> tuple[list[str], int, int | None]:
    if not log_path or not log_path.exists():
        return [], 0, None
    outputs: list[str] = []
    max_end_ms: int | None = None
    try:
        with log_path.open("r", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                try:
                    end_ms = int(parts[1])
                    max_end_ms = max(end_ms, max_end_ms or 0)
                except ValueError:
                    pass
                outputs.append(parts[3])
    except OSError:
        return [], 0, None
    return outputs, len(set(outputs)), max_end_ms


def count_build_edges(ninja_path: Path | None) -> int | None:
    if not ninja_path or not ninja_path.exists():
        return None
    count = 0
    try:
        with ninja_path.open("r", errors="replace") as f:
            for line in f:
                if line.startswith("build "):
                    count += 1
    except OSError:
        return None
    return count


def newest_files(paths: Iterable[Path], patterns: Iterable[str], limit: int = 5) -> list[Path]:
    found: list[Path] = []
    for base in paths:
        if not base.exists():
            continue
        for pattern in patterns:
            try:
                if base.is_dir():
                    found.extend(base.rglob(pattern))
                elif base.match(pattern):
                    found.append(base)
            except OSError:
                continue
    return sorted((p for p in found if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


def wheel_candidates(root: Path) -> list[Path]:
    bases = [root]
    try:
        bases.extend(p for p in root.glob("dist*") if p.is_dir())
    except OSError:
        pass
    # Search dist folders recursively, but only the root top-level non-recursively.
    found: list[Path] = []
    try:
        found.extend(p for p in root.glob("*.whl") if p.is_file())
    except OSError:
        pass
    for base in bases[1:]:
        try:
            found.extend(base.rglob("*.whl"))
        except OSError:
            pass
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)[:5]


def cpu_delta_by_pid(before: list[Proc], after: list[Proc]) -> dict[int, float]:
    before_cpu = {p.pid: p.cpu_seconds for p in before}
    delta: dict[int, float] = {}
    for p in after:
        old = before_cpu.get(p.pid)
        delta[p.pid] = max(0.0, p.cpu_seconds - old) if old is not None else 0.0
    return delta


def print_status(args: argparse.Namespace) -> None:
    root = Path(args.root)
    temp_dir = Path(args.temp_dir)
    lib_dir = Path(args.lib_dir)
    sample_seconds = max(0.0, float(args.sample_seconds))

    ninja_path = find_build_file(temp_dir, "build.ninja")
    log_path = find_build_file(temp_dir, ".ninja_log")

    before = select_build_tree(iter_procs(), root, temp_dir, lib_dir)
    if sample_seconds:
        time.sleep(sample_seconds)
    after = select_build_tree(iter_procs(), root, temp_dir, lib_dir)
    deltas = cpu_delta_by_pid(before, after)

    outputs, completed_unique, max_end_ms = parse_ninja_log(log_path)
    total_edges = count_build_edges(ninja_path)
    completed_for_pct = completed_unique
    if total_edges:
        completed_for_pct = min(completed_unique, total_edges)
        pct = completed_for_pct / total_edges * 100.0
    else:
        pct = None

    active_start = min((p.start_ts for p in after if p.name in {"python", "python3", "ninja"}), default=None)
    now = time.time()
    runtime = None
    runtime_source = "unknown"
    if active_start:
        runtime = now - active_start
        runtime_source = "active process"
    elif log_path and log_path.exists() and max_end_ms is not None:
        runtime = max_end_ms / 1000.0
        active_start = log_path.stat().st_mtime - runtime
        runtime_source = "ninja log estimate"

    by_name: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "cpu": 0.0, "rss": 0.0})
    for p in after:
        group = by_name[p.name]
        group["count"] += 1
        group["cpu"] += deltas.get(p.pid, 0.0)
        group["rss"] += p.rss_mb

    whls = wheel_candidates(root)
    libs = newest_files([lib_dir], ["*.so", "*.pyd"], limit=5)

    print(f"Build status at {fmt_dt(now)}")
    print(f"Root: {root}")
    print(f"Temp: {temp_dir}")
    print(f"Lib:  {lib_dir}")
    print()
    print(f"Runtime: {fmt_duration(runtime)}")
    print(f"Started: {fmt_dt(active_start)} ({runtime_source})")
    if total_edges:
        print(f"Ninja: {completed_for_pct}/{total_edges} ({pct:.1f}%)")
    else:
        print(f"Ninja: {completed_unique} completed targets; total edge count unknown")
    if log_path and log_path.exists():
        print(f"Last Ninja log write: {fmt_dt(log_path.stat().st_mtime)}")
    else:
        print("Last Ninja log write: .ninja_log not found")

    total_cpu_delta = sum(deltas.values())
    active = bool(after)
    print()
    print(f"Build processes: {len(after)} {'active' if active else 'not running'}")
    print(f"CPU used during sample: {total_cpu_delta:.2f}s over {sample_seconds:.1f}s")
    print(
        "Compiler stages: "
        f"ptxas={sum(1 for p in after if p.name == 'ptxas')}, "
        f"cicc={sum(1 for p in after if p.name == 'cicc')}, "
        f"nvcc={sum(1 for p in after if p.name == 'nvcc')}, "
        f"ninja={sum(1 for p in after if p.name == 'ninja')}, "
        f"link/ld={sum(1 for p in after if p.name in {'ld', 'ld.lld', 'collect2'})}"
    )

    if by_name:
        print()
        print("CPU movement by process name:")
        for name, data in sorted(by_name.items(), key=lambda item: item[1]["cpu"], reverse=True):
            print(
                f"  {name:<10} count={int(data['count']):<3} "
                f"cpu+={data['cpu']:.2f}s rss={data['rss'] / 1024:.2f}GB"
            )

    if outputs:
        print()
        print("Last completed Ninja targets:")
        for target in outputs[-10:]:
            print(f"  {target}")

    print()
    if whls:
        print("Wheel output:")
        for p in whls:
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p} ({size_mb:.1f} MB, {fmt_dt(p.stat().st_mtime)})")
    else:
        print("Wheel output: none found yet")

    if libs:
        print()
        print("Newest library outputs:")
        for p in libs:
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p} ({size_mb:.1f} MB, {fmt_dt(p.stat().st_mtime)})")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show RunPod/Linux CUDA extension build progress.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Project root. Default: /workspace/flash")
    parser.add_argument("--temp-dir", default=str(DEFAULT_TEMP_DIR), help="Build temp dir containing build.ninja")
    parser.add_argument("--lib-dir", default=str(DEFAULT_LIB_DIR), help="Build lib dir containing built .so files")
    parser.add_argument("--sample-seconds", type=float, default=10.0, help="CPU sample window. Default: 10")
    parser.add_argument("--watch", type=float, default=0.0, help="Repeat every N seconds. Default: one-shot")
    parser.add_argument("--clear", action="store_true", help="Clear the terminal between --watch refreshes")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.watch and args.watch > 0:
        while True:
            if args.clear:
                print("\033c", end="")
            print_status(args)
            time.sleep(args.watch)
    else:
        print_status(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
