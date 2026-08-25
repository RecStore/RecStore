#!/usr/bin/env python3
"""Criteo Kaggle e2e 对比（RecStore-RDMA vs TorchRec-HBM），单文件版。

替代 bench_compare.sh + cluster.sh：拓扑只由两个配置给出，其余（sshpass、
preflight、清理残留进程、调 e2e CLI、检查 NCCL IB 日志）都在本文件里。

  1) 计算节点：(IP, GPU) pair 列表，第一个是 rendezvous master(node_rank=0)
     例：--clients "[(10.0.2.191, 1), (10.0.2.192, 3)]"
     或：--clients 10.0.2.191:1,10.0.2.192:3
  2) 参数服务器：IP 列表，顺序即 shard 顺序
     例：--ps "[10.0.2.191, 10.0.2.192]"
     或：--ps 10.0.2.191,10.0.2.192

不带参数运行会交互式询问这两个配置。未识别的参数原样传给
tools.benchmarks.e2e.custom.cli（如 --steps 40 --no-torchrec --dry-run）。
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CRITEO_DIR = Path(__file__).resolve().parent
REPO_ROOT = CRITEO_DIR.parents[1]
KILL_SCRIPT = REPO_ROOT / "tools/benchmarks/kill_bench_procs.sh"
PETPS_SERVER = REPO_ROOT / "build/bin/petps_server"

IB_DEVICE = "mlx5_0"
IB_CHECK = (
    f"test -e /sys/class/infiniband/{IB_DEVICE}"
    f" && grep -q ACTIVE /sys/class/infiniband/{IB_DEVICE}/ports/1/state"
    f" && grep -q InfiniBand /sys/class/infiniband/{IB_DEVICE}/ports/1/link_layer"
)
LOCAL_ALIASES = {"", "local", "localhost", "127.0.0.1", "::1"}


class ConfigError(Exception):
    pass


def log(msg: str) -> None:
    print(f"[bench_compare] {msg}", flush=True)


# ---------------------------------------------------------------- 配置解析


def _tokens(raw: str) -> list[str]:
    """把 "[(a, 1), (b, 3)]" / "a:1,b:3" 一律拆成裸 token。"""
    flat = re.sub(r"[\[\]{}()'\"]", " ", raw)
    return [tok for tok in re.split(r"[\s,]+", flat) if tok]


def parse_clients(raw: str) -> list[tuple[str, int]]:
    tokens = _tokens(raw)
    if not tokens:
        raise ConfigError("计算节点配置为空")
    pairs: list[tuple[str, int]] = []
    if any(":" in tok for tok in tokens):
        for tok in tokens:
            ip, _, gpu = tok.partition(":")
            if not ip or not gpu.isdigit():
                raise ConfigError(f"计算节点应为 IP:GPU，收到 {tok!r}")
            pairs.append((ip, int(gpu)))
    else:
        if len(tokens) % 2:
            raise ConfigError(f"计算节点应为 (IP, GPU) 成对出现，收到 {tokens}")
        for ip, gpu in zip(tokens[::2], tokens[1::2]):
            if not gpu.isdigit():
                raise ConfigError(f"GPU 卡号应为整数，收到 {gpu!r}")
            pairs.append((ip, int(gpu)))
    return pairs


def parse_ps(raw: str) -> list[str]:
    ips = _tokens(raw)
    if not ips:
        raise ConfigError("参数服务器配置为空")
    return ips


def ask(prompt: str) -> str:
    if not sys.stdin.isatty():
        raise ConfigError(f"缺少配置且不是交互终端：{prompt}")
    return input(prompt)


# ---------------------------------------------------------------- 本机地址


def local_ips() -> set[str]:
    found = set(LOCAL_ALIASES)
    for cmd in (["ip", "-o", "-4", "addr", "show"], ["ifconfig"]):
        if not shutil.which(cmd[0]):
            continue
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        found |= set(re.findall(r"inet\s+(\d+\.\d+\.\d+\.\d+)", out))
        break
    try:
        found |= set(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return found


# ---------------------------------------------------------------- ssh / 远端


class Ssh:
    """ssh 执行器；有远端主机时用 sshpass -e 的 PATH shim，密码不进命令行。"""

    def __init__(self, port: int, password: str | None) -> None:
        self.port = port
        self.password = password
        self._shim: str | None = None

    def enable_sshpass(self) -> None:
        if not self.password:
            log("SSHPASS 未设置，直接用 ssh（依赖密钥/agent）")
            return
        if not shutil.which("sshpass"):
            raise ConfigError("缺少 sshpass（或清空 SSHPASS 改用密钥登录）")
        real_ssh = shutil.which("ssh")
        if not real_ssh:
            raise ConfigError("缺少 ssh")
        self._shim = tempfile.mkdtemp(prefix="criteo-ssh-")
        shim = Path(self._shim) / "ssh"
        shim.write_text(
            "#!/bin/sh\n"
            f'exec sshpass -e {shlex.quote(real_ssh)} '
            '-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$@"\n',
            encoding="utf-8",
        )
        shim.chmod(0o700)
        os.environ["SSHPASS"] = self.password
        os.environ["PATH"] = f"{self._shim}{os.pathsep}{os.environ['PATH']}"

    def cleanup(self) -> None:
        if self._shim:
            shutil.rmtree(self._shim, ignore_errors=True)
            self._shim = None

    def _argv(self, host: str) -> list[str]:
        argv = ["ssh"]
        if self.port != 22:
            argv += ["-p", str(self.port)]
        return argv + [host]

    def bash(self, host: str, script: str, *, check: bool = True) -> int:
        """在 host 上跑一段 bash（host == "local" 时本地跑）。"""
        if host in LOCAL_ALIASES:
            cmd = ["bash", "-c", script]
        else:
            cmd = self._argv(host) + [f"bash -lc {shlex.quote(script)}"]
        proc = subprocess.run(cmd)
        if check and proc.returncode != 0:
            raise ConfigError(f"{host}: 命令失败（exit {proc.returncode}）: {script}")
        return proc.returncode

    def bash_stdin(self, host: str, body: str, *, check: bool = True) -> int:
        """把脚本内容喂给 host 上的 bash -s。"""
        cmd = ["bash", "-s"] if host in LOCAL_ALIASES else self._argv(host) + ["bash -s"]
        proc = subprocess.run(cmd, input=body, text=True)
        if check and proc.returncode != 0:
            raise ConfigError(f"{host}: bash -s 失败（exit {proc.returncode}）")
        return proc.returncode


# ---------------------------------------------------------------- preflight


def require_processed(data_dir: Path) -> None:
    if not (data_dir / "day_0_labels.npy").exists():
        raise ConfigError(
            f"缺少 {data_dir}/day_0_labels.npy；先跑：python3 {CRITEO_DIR}/preprocess.py"
        )


def require_binary() -> None:
    if not (PETPS_SERVER.is_file() and os.access(PETPS_SERVER, os.X_OK)):
        raise ConfigError(f"缺少可执行文件 {PETPS_SERVER}")


def preflight_ib(ssh: Ssh, hosts: list[str]) -> None:
    for host in hosts:
        ssh.bash(host, IB_CHECK)
    log(f"{IB_DEVICE} ACTIVE/InfiniBand ok：{', '.join(hosts)}")


def kill_stale(ssh: Ssh, hosts: list[str], *, check: bool) -> None:
    body = KILL_SCRIPT.read_text(encoding="utf-8")
    for host in hosts:
        ssh.bash_stdin(host, body, check=check)


# ---------------------------------------------------------------- 日志检查


def check_torchrec_ib(output_dir: Path) -> None:
    logs = sorted((output_dir / "outputs").rglob("torchrec_nccl_rank*.log"))
    if not logs:
        raise ConfigError(f"{output_dir}/outputs 下没有 TorchRec NCCL 日志")
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in ("NET/IB", f"{IB_DEVICE}:1/IB"):
            if needle not in text:
                raise ConfigError(f"{path} 里没有 {needle}（NCCL 没走 IB）")
    log(f"TorchRec NCCL IB ok，共 {len(logs)} 份日志")


# ---------------------------------------------------------------- 主流程


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Criteo Kaggle e2e：RecStore-RDMA vs TorchRec-HBM",
        epilog="未识别参数原样传给 tools.benchmarks.e2e.custom.cli",
    )
    parser.add_argument(
        "--clients",
        default=os.environ.get("CRITEO_CLIENTS", ""),
        help='计算节点 (IP, GPU) 列表，如 "[(10.0.2.191, 1), (10.0.2.192, 3)]"',
    )
    parser.add_argument(
        "--ps",
        default=os.environ.get("CRITEO_PS", ""),
        help='参数服务器 IP 列表，如 "[10.0.2.191, 10.0.2.192]"',
    )
    parser.add_argument("--ssh-user", default=os.environ.get("CRITEO_SSH_USER", "root"))
    parser.add_argument(
        "--ssh-port", type=int, default=int(os.environ.get("CRITEO_REMOTE_SSH_PORT", "22222"))
    )
    parser.add_argument("--ps-port", type=int, default=int(os.environ.get("CRITEO_PS_PORT", "15000")))
    parser.add_argument(
        "--master-port", type=int, default=int(os.environ.get("CRITEO_MASTER_PORT", "29500"))
    )
    parser.add_argument(
        "--data-dir", default=os.environ.get("CRITEO_PROCESSED", str(CRITEO_DIR / "processed"))
    )
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", ""))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-embeddings", type=int, default=800000)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--index-type", default="DRAM_PET_HASH")
    return parser


def resolve_topology(
    clients: list[tuple[str, int]],
    ps_ips: list[str],
    *,
    ssh_user: str,
    ssh_port: int,
    ps_port: int,
) -> tuple[list[str], list[str], list[str]]:
    """返回 (client spec, ps spec, 需要 preflight 的主机)。"""
    mine = local_ips()

    def host_of(ip: str) -> str:
        return "local" if ip in mine else f"{ssh_user}@{ip}"

    # RDMA control-plane 端口在本进程里 bind，shard0 必须落在本机。
    local_ps = [ip for ip in ps_ips if ip in mine]
    if not local_ps:
        raise ConfigError(
            "PS 列表里至少要有一台是本机（RDMA control-plane 需绑定本机地址）；"
            f"本机地址：{', '.join(sorted(ip for ip in mine if ip not in LOCAL_ALIASES))}"
        )
    head = ps_ips.index(local_ps[0])
    if head:
        ps_ips = [ps_ips[head]] + ps_ips[:head] + ps_ips[head + 1 :]
        log(f"把本机 PS {ps_ips[0]} 调到 shard0（control-plane 要求），顺序：{ps_ips}")

    client_specs = [
        f"ssh={host_of(ip)},ssh_port={ssh_port},repo={REPO_ROOT},"
        f"ip={ip},gpu={gpu},node_rank={rank},nproc=1"
        for rank, (ip, gpu) in enumerate(clients)
    ]
    ps_specs = []
    seen: dict[str, int] = {}
    for shard, ip in enumerate(ps_ips):
        # 同一台机器上多个 shard 要错开端口。
        port = ps_port + seen.get(ip, 0)
        seen[ip] = seen.get(ip, 0) + 1
        ps_specs.append(
            f"ssh={host_of(ip)},ssh_port={ssh_port},repo={REPO_ROOT},"
            f"ip={ip},port={port},shard={shard}"
        )
    hosts = list(dict.fromkeys(host_of(ip) for ip, _ in clients).keys())
    for ip in ps_ips:
        if host_of(ip) not in hosts:
            hosts.append(host_of(ip))
    return client_specs, ps_specs, hosts


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)

    clients = parse_clients(args.clients or ask("计算节点 (IP:GPU, 逗号分隔): "))
    ps_ips = parse_ps(args.ps or ask("参数服务器 IP (逗号分隔): "))
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(
        args.output_dir or REPO_ROOT / "results" / time.strftime("criteo_kaggle_e2e_%m%d%H%M")
    ).resolve()

    client_specs, ps_specs, hosts = resolve_topology(
        clients,
        ps_ips,
        ssh_user=args.ssh_user,
        ssh_port=args.ssh_port,
        ps_port=args.ps_port,
    )
    for spec in client_specs:
        log(f"client: {spec}")
    for spec in ps_specs:
        log(f"ps: {spec}")
    log(f"output-dir: {output_dir}")

    require_processed(data_dir)
    require_binary()

    # e2e CLI 调的是裸 ssh：密码走 sshpass shim，端口走 spec 里的 ssh_port。
    remote = [host for host in hosts if host not in LOCAL_ALIASES]
    ssh = Ssh(args.ssh_port, os.environ.get("SSHPASS", "1234") if remote else None)
    if remote:
        ssh.enable_sshpass()

    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO_ROOT / "src/python/pytorch"),
            str(REPO_ROOT / "src"),
            *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []),
        ]
    )
    os.environ["PYTHONUNBUFFERED"] = "1"

    cli_argv = ["--data-dir", str(data_dir)]
    for spec in client_specs:
        cli_argv += ["--client", spec]
    for spec in ps_specs:
        cli_argv += ["--ps", spec]
    cli_argv += [
        "--transports", "rdma",
        "--batch-size", str(args.batch_size),
        "--num-embeddings", str(args.num_embeddings),
        "--steps", str(args.steps),
        "--warmup-steps", str(args.warmup_steps),
        "--repeat", str(args.repeat),
        "--index-type", args.index_type,
        "--master-port", str(args.master_port),
        "--output-dir", str(output_dir),
        "--skip-build",
        "--skip-tests",
        *extra,
    ]

    try:
        preflight_ib(ssh, hosts)
        kill_stale(ssh, hosts, check=True)
        cmd =[sys.executable, "-m", "tools.benchmarks.e2e.custom.cli", *cli_argv]
        log("run: " + " ".join(shlex.quote(part) for part in cmd))
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        if rc != 0:
            log(f"e2e CLI 失败（exit {rc}）")
            return rc
        if "--dry-run" not in extra and "--no-torchrec" not in extra:
            check_torchrec_ib(output_dir)
        log(f"summary: {output_dir / 'summary.md'}")
    finally:
        kill_stale(ssh, hosts, check=False)
        ssh.cleanup()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as err:
        print(f"[bench_compare] {err}", file=sys.stderr)
        raise SystemExit(1)
