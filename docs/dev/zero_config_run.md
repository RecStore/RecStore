# Docker 开发镜像

为了方便开发者快速体验 RecStore 而无需在本地搭建完整编译环境，我们现在提供基于 Docker 镜像的免配置运行方式。需要从源码构建时见 [从源码构建](../quickstart.md)。

推荐直接使用已经构建好的开发镜像：

- 包页面：<https://github.com/Choimoe/RecStore/pkgs/container/recstore%2Frecstore-devel>
- 镜像地址：`ghcr.io/choimoe/recstore/recstore-devel`

???+ Warning "注意"
    该方式主要面向 Linux 环境。

    当前公开镜像默认是 CPU 开发环境，主要用于本地调试、单机测试和快速验证，不等同于生产部署方案。

通过直接拉取云端构建好的 Docker 镜像，你可以在标准 Linux 环境（如 Ubuntu 20.04/22.04）中快速进入可运行的 RecStore 环境。

## 1. 获取镜像

### 1.1 在网页上查看镜像

打开 GitHub Packages 页面：

- <https://github.com/Choimoe/RecStore/pkgs/container/recstore%2Frecstore-devel>

你可以在这里查看镜像说明、可用 tag，以及最近的发布时间。

### 1.2 使用 Docker 拉取镜像

通常直接拉取 `latest` 即可：

```bash
docker pull ghcr.io/choimoe/recstore/recstore-devel:latest
```

如果你需要固定某个版本，也可以把 `latest` 替换为具体 tag。

## 2. 环境准备

宿主机只需要准备 Docker 即可，不再要求本地提前安装编译依赖。

### 2.1 准备代码仓库

建议克隆仓库，并把代码目录挂载进容器。这样可以直接复用仓库里的配置、脚本和示例：

```bash
git clone --depth 1 --single-branch https://github.com/RecStore/RecStore.git
cd RecStore
```

### 2.2 安装 Docker

请确保宿主机已经安装可用的 Docker 环境：

```bash
docker --version
```

如果这里命令不可用，请先按你的发行版方式安装 Docker。

## 3. 启动容器环境

### 3.1 以交互方式启动容器

下面的命令会把当前仓库挂载到容器内的 `/workspace`，并保持容器常驻：

```bash
docker run -it --rm \
  --name recstore-dev \
  --network host \
  -v "$(pwd):/workspace" \
  ghcr.io/choimoe/recstore/recstore-devel:latest \
  bash
```

进入容器后：

```bash
cd /workspace
```

### 3.2 以后台方式启动容器

如果你希望容器在后台运行，方便多次 `docker exec` 进入，可以这样启动：

```bash
docker run -d \
  --name recstore-dev \
  --network host \
  -v "$(pwd):/workspace" \
  ghcr.io/choimoe/recstore/recstore-devel:latest \
  bash -lc 'tail -f /dev/null'
```

之后进入容器：

```bash
docker exec -it recstore-dev bash
cd /workspace
```

## 4. 容器内运行 RecStore

### 4.1 构建项目

如果镜像已经包含初始化后的开发环境，通常只需要在容器内执行常规构建：

```bash
cd /workspace
mkdir -p build
cd build
cmake -DENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make -j4
```

### 4.2 启动参数服务器

构建完成后，可以直接启动 `ps_server`：

```bash
cd /workspace
./build/bin/ps_server --config_path ./recstore_config.json
```

如果要在另一个终端进入同一个容器继续调试：

```bash
docker exec -it recstore-dev bash
cd /workspace
```

### 4.3 运行测试

例如运行 CTest：

```bash
cd /workspace/build
ctest --output-on-failure
```

或者运行 PyTorch client 相关测试：

```bash
cd /workspace/build
ctest -R pytorch_client_test -VV
```

## 5. 常见说明

- 这个页面讲的是“直接使用已构建 Docker 环境”，不是下载二进制 tar 包后在宿主机手动拼运行时依赖。
- 如果你只是想快速进入一个可工作的 RecStore 开发环境，优先使用 `ghcr.io/choimoe/recstore/recstore-devel`。
- 如果你需要固定环境，请显式指定镜像 tag，而不是依赖 `latest`。
