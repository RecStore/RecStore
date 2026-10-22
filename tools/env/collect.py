#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derived from vLLM's vllm/collect_env.py (Apache-2.0), which in turn borrows
# from PyTorch's torch/utils/collect_env.py (BSD-3-Clause). Modified for
# RecStore: RDMA/NIC, GID table, NCCL/GDRCopy, build-tree, and env-var
# collection plus --redact masking; several helpers (run, run_and_read_all,
# run_and_parse_first_match, pretty_str internals) are taken near-verbatim.
"""Collect environment information for RecStore bug reports and benchmark
reproduction.

Usage:
    python tools/env/collect.py            # print to stdout
    python tools/env/collect.py > env.txt  # save alongside a report
    python tools/env/collect.py --redact   # mask host-identifying values

The script is stdlib-only and never crashes on missing tools: any value it
cannot collect is reported as "Could not collect".
"""

import datetime
import os
import re
import subprocess
import sys
from collections import namedtuple

try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

COMMAND_TIMEOUT_S = 30

# Repository root (this file lives at <root>/tools/env/collect.py).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# System Environment Information
SystemEnv = namedtuple(
    "SystemEnv",
    [
        # RecStore
        "recstore_version",
        "git_branch",
        "git_hash",
        "git_dirty",
        "build_info",
        # System
        "os",
        "kernel",
        "gcc_version",
        "clang_version",
        "cmake_version",
        "libc_version",
        # Python
        "python_version",
        "python_platform",
        "pip_version",
        "pip_packages",
        "conda_packages",
        # CUDA / GPU
        "torch_version",
        "is_cuda_available",
        "cuda_compiled_version",
        "cuda_runtime_version",
        "nvidia_gpu_models",
        "nvidia_driver_version",
        "cudnn_version",
        "nccl_version",
        "is_gdr_copy_available",
        "gpu_topo",
        # RDMA / NIC
        "ofed_version",
        "rdma_kernel_driver",
        "ibv_devices",
        "ibv_devinfo",
        # CPU / memory
        "cpu_info",
        "mem_info",
        # Environment variables
        "env_vars",
    ],
)

DEFAULT_PIP_PATTERNS = {
    "torch",
    "numpy",
    "nvidia",
    "nccl",
    "recstore",
    "transformers",
    "pybind",
    "gdr",
}

DEFAULT_CONDA_PATTERNS = {
    "torch",
    "numpy",
    "nccl",
    "mkl",
    "nvidia",
    "recstore",
}

# Environment variables worth reporting (prefix match), mirroring the env
# knobs RecStore, NCCL and the CUDA/RDMA stack actually read.
ENV_REPORT_PREFIXES = (
    "RECSTORE",
    "NCCL",
    "CUDA",
    "CUDNN",
    "TORCH",
    "PYTORCH",
    "LD_",
    "LIBRARY_PATH",
    "IBV",
    "RDMA",
    "MLX",
    "GLOG",
    "GFLAGS",
    "BRPC",
    "PS_",
    "UCX",
    "OFED",
    "OMPI",
    "AWS_",
)

# Substrings that mark an environment variable as sensitive; never print them.
SECRET_TERMS = ("secret", "token", "api", "access", "password", "passwd", "key_file")


def run(command):
    """Return (return-code, stdout, stderr); never raises on missing binaries."""
    shell = True if type(command) is str else False
    try:
        p = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell
        )
    except FileNotFoundError:
        cmd_str = command if isinstance(command, str) else command[0]
        return 127, "", "Command not found: {}".format(cmd_str)
    try:
        raw_output, raw_err = p.communicate(timeout=COMMAND_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        cmd_str = command if isinstance(command, str) else command[0]
        return 124, "", "Command timed out: {}".format(cmd_str)
    rc = p.returncode
    output = raw_output.decode("utf-8", errors="replace").strip()
    err = raw_err.decode("utf-8", errors="replace").strip()
    return rc, output, err


def run_and_read_all(run_lambda, command):
    """Run command using run_lambda; reads and returns entire output if rc is 0."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def run_and_parse_first_match(run_lambda, command, regex):
    """Run command using run_lambda, returns the first regex match if it exists."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def get_platform():
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("darwin"):
        return "darwin"
    return sys.platform


# ---------------------------------------------------------------------------
# RecStore info
# ---------------------------------------------------------------------------


def get_recstore_version():
    version_file = os.path.join(REPO_ROOT, "VERSION")
    try:
        with open(version_file) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _git(run_lambda, *args):
    return run_and_read_all(
        run_lambda, ["git", "-C", REPO_ROOT] + list(args)
    )


def get_git_info(run_lambda):
    branch = _git(run_lambda, "rev-parse", "--abbrev-ref", "HEAD")
    commit = _git(run_lambda, "rev-parse", "--short", "HEAD")
    if branch is None and commit is None:
        return None, None, None
    status = _git(run_lambda, "status", "--porcelain")
    dirty = bool(status)
    return branch, commit, dirty


# CMakeCache entries worth surfacing in a bug report.
CMAKE_CACHE_KEYS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_CXX_COMPILER",
    "CMAKE_CUDA_COMPILER",
    "ENABLE_CUDA",
    "USE_",
    "RECSTORE_",
)


def _collect_build_tree(run_lambda, path):
    if not os.path.isdir(path):
        return None
    lines = []
    if os.path.isfile(os.path.join(path, "CMakeCache.txt")):
        try:
            with open(os.path.join(path, "CMakeCache.txt")) as f:
                for line in f:
                    if ":INTERNAL=" in line:
                        continue
                    if line.startswith(CMAKE_CACHE_KEYS) and not line.endswith("=\n"):
                        lines.append(line.rstrip())
        except OSError:
            pass
    # Timestamp of the server binary, if it has been built.
    for candidate in ("bin/ps_server", "bin/ps_server_test"):
        binary = os.path.join(path, candidate)
        if os.path.isfile(binary):
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(binary))
            lines.append("{} built at {}".format(candidate, mtime))
            break
    if not lines:
        lines.append("CMake cache not found and no ps_server binary built")
    return "\n".join(lines)


def get_build_info(run_lambda):
    """Summarize the local CMake build trees (build/, build_release/)."""
    sections = []
    for name in ("build", "build_release"):
        path = os.path.join(REPO_ROOT, name)
        content = _collect_build_tree(run_lambda, path)
        if content is None:
            continue
        sections.append("[{}] \n{}".format(name, content))
    return "\n\n".join(sections) if sections else None


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


def get_os(run_lambda):
    from platform import machine

    if get_platform() == "linux":
        desc = run_and_parse_first_match(
            run_lambda, "cat /etc/*-release", r'PRETTY_NAME="(.*)"'
        )
        if desc is None:
            desc = run_and_parse_first_match(
                run_lambda, "lsb_release -a", r"Description:\t(.*)"
            )
        if desc is not None:
            return "{} ({})".format(desc, machine())
        return "{} ({})".format(get_platform(), machine())
    return get_platform()


def get_kernel():
    import platform

    if get_platform() != "linux":
        return "N/A"
    return platform.release()


def get_gcc_version(run_lambda):
    # "gcc (GCC) 11.5.0" / "gcc (Ubuntu 11.4.0-...) 11.4.0"
    version = run_and_parse_first_match(
        run_lambda, "gcc --version", r"gcc \([^)]*\) (.*)"
    )
    if version is None:
        version = run_and_parse_first_match(run_lambda, "gcc --version", r"gcc (.*)")
    return version


def get_clang_version(run_lambda):
    version = run_and_parse_first_match(
        run_lambda, "clang --version", r"clang version \([^)]*\) (.*)"
    )
    if version is None:
        version = run_and_parse_first_match(
            run_lambda, "clang --version", r"clang version (.*)"
        )
    return version


def get_cmake_version(run_lambda):
    version = run_and_parse_first_match(
        run_lambda, "cmake --version", r"cmake version (.*)"
    )
    if version is None:
        version = run_and_parse_first_match(run_lambda, "cmake --version", r"cmake (.*)")
    return version


def get_libc_version():
    import platform

    if get_platform() != "linux":
        return "N/A"
    return "-".join(platform.libc_ver())


def get_cpu_info(run_lambda):
    if get_platform() == "linux":
        rc, out, err = run_lambda("lscpu")
    else:
        rc, out, err = 1, "", "Only supported on Linux"
    return out if rc == 0 else err


def get_mem_info():
    """Memory and hugepage summary from /proc/meminfo (hugepages matter for RDMA)."""
    if get_platform() != "linux":
        return "N/A"
    wanted = (
        "MemTotal",
        "MemAvailable",
        "SwapTotal",
        "HugePages_Total",
        "Hugepagesize",
    )
    try:
        with open("/proc/meminfo") as f:
            lines = [line.rstrip() for line in f if line.startswith(wanted)]
    except OSError:
        return None
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CUDA / GPU info
# ---------------------------------------------------------------------------


def get_nvidia_smi():
    return "nvidia-smi"


def get_gpu_info(run_lambda):
    import re

    rc, out, _ = run_lambda(get_nvidia_smi() + " -L")
    if rc != 0:
        return None
    # Anonymize GPUs by removing their UUID
    return re.sub(r" \(UUID: .+?\)", "", out)


def get_nvidia_driver_version(run_lambda):
    return run_and_parse_first_match(
        run_lambda, get_nvidia_smi(), r"Driver Version: (.*?) "
    )


def get_running_cuda_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "nvcc --version", r"release .+ V(.*)")


def get_cudnn_version(run_lambda):
    """Return a list of libcudnn.so; it's hard to tell which one is being used."""
    rc, out, _ = run_lambda('ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev')
    if len(out) == 0 or rc not in (0, 1):
        return None
    files_set = set()
    for fn in out.split("\n"):
        fn = os.path.realpath(fn)  # eliminate symbolic links
        if os.path.isfile(fn):
            files_set.add(fn)
    if not files_set:
        return None
    files = sorted(files_set)
    if len(files) == 1:
        return files[0]
    return "Probably one of the following:\n{}".format("\n".join(files))


def get_nccl_version(run_lambda):
    """libnccl found by the dynamic loader; the pip package (if any) is listed
    separately under pip packages."""
    rc, out, _ = run_lambda('ldconfig -p | grep libnccl.so | rev | cut -d" " -f1 | rev')
    if rc != 0 or not out:
        return None
    files = sorted({os.path.realpath(fn) for fn in out.split("\n") if fn})
    return "\n".join(files)


def get_is_gdr_copy_available():
    """GDRCopy is used by the RDMA path for GPU-direct transfers."""
    device = os.path.exists("/dev/gdrdrv")
    rc, out, _ = run('ldconfig -p | grep libgdrapi')
    lib = rc == 0 and bool(out)
    if device and lib:
        return True
    if device or lib:
        return "Partial (/dev/gdrdrv: {}, libgdrapi: {})".format(device, lib)
    return False


def get_gpu_topo(run_lambda):
    if get_platform() == "linux":
        return run_and_read_all(run_lambda, "nvidia-smi topo -m")
    return None


# ---------------------------------------------------------------------------
# RDMA / NIC info
# ---------------------------------------------------------------------------


def get_ofed_version(run_lambda):
    """OFED / rdma-core stack version."""
    version = run_and_parse_first_match(
        run_lambda, "ofed_info -s", r"(MLNX_OFED_LINUX-\S+|rdma-core\S*)"
    )
    if version is not None:
        return version
    # Inbox stacks: ask the package manager instead.
    for cmd in ("rpm -q rdma-core", "dpkg -s rdma-core"):
        out = run_and_read_all(run_lambda, cmd)
        if out:
            return out.splitlines()[0]
    return None


def get_rdma_kernel_driver(run_lambda):
    """Kernel driver version for each RDMA device reported by ibv_devices."""
    rc, out, _ = run_lambda("ibv_devices")
    if rc != 0:
        return None
    devices = [line.split()[0] for line in out.splitlines()[2:] if line.strip()]
    if not devices:
        return None
    results = []
    for dev in devices:
        # mlx5_3 -> mlx5_core, hns_0 -> hns, irdma0 stays irdma0, ...
        base = dev.rstrip("0123456789").rstrip("_")
        for module in (base + "_core", base, dev):
            rc, out, _ = run_lambda("modinfo {}".format(module))
            match = (
                re.search(r"^version:\s+(.*)", out, re.MULTILINE) if rc == 0 else None
            )
            if match is not None:
                results.append("{} ({}): {}".format(dev, module, match.group(1)))
                break
        else:
            results.append("{}: module version unknown".format(dev))
    return "\n".join(results)


def get_ibv_devices(run_lambda):
    return run_and_read_all(run_lambda, "ibv_devices")


def get_ibv_devinfo(run_lambda):
    """Verbose device info: port state, link layer (RoCE v1/v2, IB) and the
    full GID table — the gid_index questions come up in almost every RDMA
    bug report, so include it verbatim."""
    return run_and_read_all(run_lambda, "ibv_devinfo -v")


# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------


def get_python_platform():
    import platform

    return platform.platform()


def get_pip_packages(run_lambda, patterns=None):
    """Return `pip list` output. Note: will also find conda-installed packages."""
    if patterns is None:
        patterns = DEFAULT_PIP_PATTERNS
    cmd = [sys.executable, "-mpip", "list", "--format=freeze"]
    out = run_and_read_all(run_lambda, cmd)
    if out is None:
        pip_version, out = None, None
    else:
        pip_version = "pip3" if sys.version[0] == "3" else "pip"
        out = "\n".join(
            line for line in out.splitlines() if any(name in line for name in patterns)
        )
    return pip_version, out


def get_conda_packages(run_lambda, patterns=None):
    if patterns is None:
        patterns = DEFAULT_CONDA_PATTERNS
    conda = os.environ.get("CONDA_EXE", "conda")
    out = run_and_read_all(run_lambda, [conda, "list"])
    if out is None:
        return out
    return "\n".join(
        line
        for line in out.splitlines()
        if not line.startswith("#") and any(name in line for name in patterns)
    )


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------


def get_env_vars():
    env_vars = ""
    for k, v in os.environ.items():
        if any(term in k.lower() for term in SECRET_TERMS):
            continue
        if k.startswith(ENV_REPORT_PREFIXES):
            env_vars += "{}={}\n".format(k, v)
    return env_vars


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def get_env_info():
    run_lambda = run
    pip_version, pip_list_output = get_pip_packages(run_lambda)

    if TORCH_AVAILABLE:
        version_str = torch.__version__
        cuda_available_str = str(torch.cuda.is_available())
        cuda_version_str = torch.version.cuda
    else:
        version_str = cuda_available_str = cuda_version_str = "N/A"

    git_branch, git_hash, git_dirty = get_git_info(run_lambda)

    return SystemEnv(
        recstore_version=get_recstore_version(),
        git_branch=git_branch,
        git_hash=git_hash,
        git_dirty=git_dirty,
        build_info=get_build_info(run_lambda),
        os=get_os(run_lambda),
        kernel=get_kernel(),
        gcc_version=get_gcc_version(run_lambda),
        clang_version=get_clang_version(run_lambda),
        cmake_version=get_cmake_version(run_lambda),
        libc_version=get_libc_version(),
        python_version="{} ({}-bit runtime)".format(
            sys.version.replace("\n", " "), sys.maxsize.bit_length() + 1
        ),
        python_platform=get_python_platform(),
        pip_version=pip_version,
        pip_packages=pip_list_output,
        conda_packages=get_conda_packages(run_lambda),
        torch_version=version_str,
        is_cuda_available=cuda_available_str,
        cuda_compiled_version=cuda_version_str,
        cuda_runtime_version=get_running_cuda_version(run_lambda),
        nvidia_gpu_models=get_gpu_info(run_lambda),
        nvidia_driver_version=get_nvidia_driver_version(run_lambda),
        cudnn_version=get_cudnn_version(run_lambda),
        nccl_version=get_nccl_version(run_lambda),
        is_gdr_copy_available=get_is_gdr_copy_available(),
        gpu_topo=get_gpu_topo(run_lambda),
        ofed_version=get_ofed_version(run_lambda),
        rdma_kernel_driver=get_rdma_kernel_driver(run_lambda),
        ibv_devices=get_ibv_devices(run_lambda),
        ibv_devinfo=get_ibv_devinfo(run_lambda),
        cpu_info=get_cpu_info(run_lambda),
        mem_info=get_mem_info(),
        env_vars=get_env_vars(),
    )


env_info_fmt = """
==============================
        System Info
==============================
OS                           : {os}
Kernel                       : {kernel}
GCC version                  : {gcc_version}
Clang version                : {clang_version}
CMake version                : {cmake_version}
Libc version                 : {libc_version}

==============================
       RecStore Info
==============================
RecStore Version             : {recstore_version}
Git branch                   : {git_branch}
Git commit                   : {git_hash}
Git working tree dirty       : {git_dirty}
Build trees:
  {build_info}

==============================
       CUDA / GPU Info
==============================
PyTorch version              : {torch_version}
Is CUDA available            : {is_cuda_available}
CUDA used to build PyTorch   : {cuda_compiled_version}
CUDA runtime version         : {cuda_runtime_version}
GPU models and configuration : {nvidia_gpu_models}
Nvidia driver version        : {nvidia_driver_version}
cuDNN version                : {cudnn_version}
NCCL version                 : {nccl_version}
GDRCopy available            : {is_gdr_copy_available}
GPU Topology:
  {gpu_topo}

==============================
      RDMA / NIC Info
==============================
OFED / rdma-core version     : {ofed_version}
RDMA kernel driver           : {rdma_kernel_driver}
RDMA devices (ibv_devices)   :
  {ibv_devices}
RDMA device info (ibv_devinfo -v):
  {ibv_devinfo}

==============================
          CPU Info
==============================
{cpu_info}

==============================
          Memory Info
==============================
{mem_info}

==============================
Versions of relevant libraries
==============================
{pip_packages}
{conda_packages}

==============================
     Environment Variables
==============================
{env_vars}
""".strip()


def pretty_str(envinfo):
    def replace_nones(dct, replacement="Could not collect"):
        for key in dct.keys():
            if dct[key] is None:
                dct[key] = replacement
        return dct

    def replace_bools(dct, true="Yes", false="No"):
        for key in dct.keys():
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    def prepend(text, tag="[prepend]"):
        return "\n".join(tag + line for line in text.split("\n"))

    def replace_if_empty(text, replacement="No relevant packages"):
        if text is not None and len(text) == 0:
            return replacement
        return text

    def maybe_start_on_next_line(string):
        # If `string` is multiline, prepend a \n to it.
        if string is not None and len(string.split("\n")) > 1:
            return "\n{}\n".format(string)
        return string

    mutable_dict = envinfo._asdict()

    # Replace True with Yes, False with No
    mutable_dict = replace_bools(mutable_dict)

    # Replace all None objects with 'Could not collect'
    mutable_dict = replace_nones(mutable_dict)

    # If either of these are '', replace with 'No relevant packages'
    mutable_dict["pip_packages"] = replace_if_empty(mutable_dict["pip_packages"])
    mutable_dict["conda_packages"] = replace_if_empty(mutable_dict["conda_packages"])

    # Tag conda and pip packages with a prefix
    # If they were previously None, they'll show up as ie '[conda] Could not collect'
    if mutable_dict["pip_packages"]:
        mutable_dict["pip_packages"] = prepend(
            mutable_dict["pip_packages"], "[{}] ".format(envinfo.pip_version)
        )
    if mutable_dict["conda_packages"]:
        mutable_dict["conda_packages"] = prepend(
            mutable_dict["conda_packages"], "[conda] "
        )

    # Multiline fields read better starting on their own line
    for key in (
        "nvidia_gpu_models",
        "nccl_version",
        "cudnn_version",
        "gpu_topo",
        "rdma_kernel_driver",
        "ibv_devices",
        "ibv_devinfo",
        "env_vars",
    ):
        mutable_dict[key] = maybe_start_on_next_line(mutable_dict.get(key))

    return env_info_fmt.format(**mutable_dict)


def get_pretty_env_info():
    return pretty_str(get_env_info())


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

# Internal-only path prefixes that should not appear in a public report.
INTERNAL_PATH_PREFIXES = (
    "/opt/meituan",
)


def _mask_gid_value(value):
    """Mask the host-specific half of a GID, keeping the type-distinguishing
    prefix recognizable: link-local (fe80::/64, whose lower half is MAC-derived,
    so mask all of it) vs IPv4-mapped RoCE v2 (::ffff:x.x.x.x) vs global IB
    subnets (mask only the host part)."""
    parts = value.split(":")
    if len(parts) != 8:
        return value
    keep = 4 if parts[0].lower() == "fe80" else 6
    return ":".join(parts[:keep] + ["xxxx"] * (8 - keep))


def _mask_ipv4(m):
    """Keep the first two octets (subnet still diagnosable), mask the rest."""
    return "{}.x.x".format(m.group(1))


def redact_text(text):
    """Mask host- and site-identifying values while keeping the output useful
    for debugging: GID/IP/MAC/GUID addresses, home directory, internal paths."""
    # GID table entries in `ibv_devinfo -v` output
    text = re.sub(
        r"(GID\[\s*\d+\]:\s*)([0-9a-fA-F:]+)",
        lambda m: m.group(1) + _mask_gid_value(m.group(2)),
        text,
    )
    # node_guid / sys_image_guid in `ibv_devinfo -v` output
    text = re.sub(
        r"((?:node_guid|sys_image_guid):\s*)([0-9a-fA-F:]+)",
        lambda m: m.group(1) + ":".join(m.group(2).split(":")[:2]) + ":xx:xx:xx:xx",
        text,
    )
    # Dotted IPv4 addresses anywhere (env vars included). `==` lookbehind
    # keeps pip package versions like `nvidia-curand-cu12==10.3.0.727` intact.
    text = re.sub(
        r"(?<!==)\b(\d{1,3}\.\d{1,3})\.\d{1,3}\.\d{1,3}\b", _mask_ipv4, text
    )
    # Colon-separated MAC addresses, keeping the OUI (vendor) bytes
    text = re.sub(
        r"\b[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}\b",
        lambda m: ":".join(m.group(0).split(":")[:3] + ["xx", "xx", "xx"]),
        text,
    )
    # ibv_devices rows: "<device>    <16-hex-digit node GUID>" (no colons)
    text = re.sub(
        r"^(\s*\S+\s+)([0-9a-fA-F]{16})\s*$",
        lambda m: m.group(1) + m.group(2)[:8] + "xxxxxxxx",
        text,
        flags=re.MULTILINE,
    )
    # Home directory paths -> /home/<user>
    home = os.path.expanduser("~")
    if home and home != "~":
        text = text.replace(home, "/home/<user>")
    # Internal-only path prefixes
    for prefix in INTERNAL_PATH_PREFIXES:
        text = text.replace(prefix, os.path.dirname(prefix) + "/<internal>")
    return text


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect RecStore environment information for bug reports."
    )
    parser.add_argument(
        "--redact",
        action="store_true",
        help="mask host-identifying values (IPs, MACs, GIDs, GUIDs, home "
        "directory, internal paths) before printing, for public reports",
    )
    args = parser.parse_args()

    print("Collecting environment information...")
    output = get_pretty_env_info()
    if args.redact:
        print("(redaction enabled: IPs/MACs/GIDs masked, home dir anonymized)")
        output = redact_text(output)
    print(output)


if __name__ == "__main__":
    main()
