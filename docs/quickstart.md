# 从源码构建

!!! note "待重写"

    与 Docker 开发镜像页面合并重复的环境准备步骤。

本指南将帮助你搭建 RecStore 的开发和运行环境，相关步骤出现错误可以查看 [常见问题](./faq.md)。

## 0. 环境准备

RecStore 推荐使用 Docker 进行环境配置。在开始之前，请确保你的系统已安装以下工具：

*   **Docker**: [安装指南](https://docs.docker.com/engine/install/ubuntu/)
*   **NVIDIA Docker (NVIDIA Container Toolkit)**: [安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

??? info "Ubuntu 快速安装脚本"

    如果你使用的是 Ubuntu 系统，可以使用以下命令快速安装 Docker 和 NVIDIA Container Toolkit：

    ```bash
    # 1. 安装 Docker
    curl -fsSL https://get.docker.com | sudo sh

    # 2. 安装 NVIDIA Container Toolkit
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    ```

然后获取 RecStore 仓库并更新子模块（取决于网络状况，首次执行拉取子模块可能耗时较长，请耐心等待）：

```bash
git clone --depth 1 --single-branch https://github.com/RecStore/RecStore.git
cd RecStore
mapfile -t submodules < <(
    git config -f .gitmodules --get-regexp path |
        awk '$2 != "third_party/faster" && $2 != "third_party/HugeCTR" { print $2 }'
)
git submodule update --init --recursive -- "${submodules[@]}"
git config core.hooksPath .githooks  # 启用仓库自带的提交前格式化检查
```

??? note "下载打包后的子模块"
    如果你希望一次性下载第三方依赖，也可以访问 [Release](https://github.com/RecStore/RecStore/releases) 页面，下载发布产物中的 `recstore-<version>-linux-x86_64-third_party.tar.gz`。该压缩包是对 `third_party` 目录的完整打包，解压到 RecStore 根目录即可：

    ```bash
    # in RecStore root directory
    tar -xzvf recstore-<version>-linux-x86_64-third_party.tar.gz -C .
    ```


## 1. 构建 Docker 镜像

进入 `dockerfiles` 目录并构建镜像：

```bash
cd dockerfiles
sudo docker build -f Dockerfile.recstore --build-arg uid=$UID  -t recstore .
cd -
```

## 2. 启动容器

=== "一键启动"

    你可以使用我们提供的脚本（按照需要修改脚本内的变量）：

    ```bash
    cd dockerfiles
    bash start_docker.sh
    cd -
    ```

=== "手动启动"

    **请务必根据你的实际环境修改路径映射 ( `-v` 选项)**。

    ```bash linenums="1" hl_lines="5-8"
    RECSTORE_PATH="$(cd .. && pwd)" \
    sudo docker run --cap-add=SYS_ADMIN --privileged \
        --security-opt seccomp=unconfined --runtime=nvidia \
        --name recstore --net=host \
        -v ${RECSTORE_PATH}:${RECSTORE_PATH} \
        -v /dev/shm:/dev/shm \
        -v /dev/hugepages:/dev/hugepages \
        -v /dev:/dev \
        -w ${RECSTORE_PATH} \
        --rm -it --gpus all -d recstore
    ```

进入容器：

```bash
sudo docker exec -it recstore /bin/bash
```

## 3. 容器内环境初始化

进入容器后，运行以下脚本进行一键初始化：

```bash title="初始化环境依赖"
cd dockerfiles
bash init_env_inside_docker.sh > init_env.log 2>&1
```

FASTERKV 和 HPS 默认不安装。如需使用，可增量执行：

```bash
bash init_env_inside_docker.sh --third-party-engines > init_engines.log 2>&1
```

```bash title="初始化 PyTorch 依赖"
# in RecStore/dockerfiles directory
bash init_dlrm.sh --mirror=0
```

## 4. 编译 RecStore

最后，编译项目：

```bash
# in RecStore root directory
mkdir -p build && cd build
```

```bash title="编译 RecStore"
# in RecStore/build directory
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.10 ..
make -j
```

Release 默认构建测试，关闭性能上报和第三方存储引擎。启用 FASTERKV 和 HPS 时，增加：

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
    -DRECSTORE_ENABLE_THIRD_PARTY_ENGINES=ON ..
make -j
```

## 环境信息

??? note "PyTorch 开发版本"
    当前推荐的 GPU 开发环境版本为：

    - `torch==2.7.1+cu118`
    - `torchrec==1.2.0`
    - `fbgemm-gpu==1.2.0`
    - `torchmetrics==1.0.3`

    其中 `torch` 需要满足 `torch.compiled_with_cxx11_abi() == True`。`dockerfiles/init_dlrm.sh` 默认会复用已经安装好的全局 `torch`，并安装与其匹配的 TorchRec 依赖。

    如需手动安装，可直接执行：

    ```bash
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.7.1
    python3 -m pip install torchrec==1.2.0 fbgemm-gpu==1.2.0 torchmetrics==1.0.3 tqdm
    ```

??? note "CI 环境说明"
    CI 当前仍以 CPU-only 为主，用于构建与基础验证。CI 中会继续安装 CPU 版 `torch` 和 `libtorch`，但在无 CUDA toolkit 或 CPU-only `torch` 环境下，`dockerfiles/init_dlrm.sh` 会自动跳过 TorchRec/FBGEMM GPU 依赖安装。
