# 常见问题 (FAQ)

## Env Issues

??? failure "ERROR: python3 is missing modules: elftools"
    **运行 `init_env_inside_docker.sh` 时出错：**

    ```text
    buildtools/meson.build:14:31: ERROR: python3 is missing modules: elftools
    ```

    虚拟环境没有退出，编译 spdk 的时候找到了虚拟环境的 `elftools`，退出 docker 重进就好了。

## Network Issues

??? failure "installing go v1.21.1 to /opt/go/1.21.1/bin gzip: stdin: unexpected end of file"
    **运行 `init_env_inside_docker.sh` 时出错：**

    ```text
    <RecStore Path>/third_party/spdk/scripts/pkgdep/common.sh: line 210: go: command not found
    installing go v1.21.1 to /opt/go/1.21.1/bin

    gzip: stdin: unexpected end of file
    tar: Child returned status 1
    tar: Error is not recoverable: exiting now
    ```

    下不下来 `go` 安装包，手动下载然后 `sudo tar -C /opt/go -xzf go1.21.1.linux-amd64.tar.gz` 再 `export PATH=$PATH:/opt/go/go/bin` 就好了。

??? failure "error: RPC failed; curl 56 GnuTLS recv error (-54): Error in the pull function."
    **运行 `init_env_inside_docker.sh` 时出错：**

    ```text
    Cloning into '/usr/src/markdownlint'...
    error: RPC failed; curl 56 GnuTLS recv error (-54): Error in the pull function.
    error: 1176 bytes of body are still expected
    fetch-pack: unexpected disconnect while reading sideband packet
    fatal: early EOF
    fatal: fetch-pack: invalid index-pack output
    ```

    可以手动 clone：

    ```bash
    git clone git@github.com:DavidAnson/markdownlint.git /usr/src/markdownlint
    ```

## CMake Errors

??? failure "By not providing 'Findcpptrace.cmake' in CMAKE_MODULE_PATH this project has asked CMake to find a package configuration file provided by 'cpptrace'"
    **cmake 编译时出错：**

    ```text
    CMake Error at src/base/CMakeLists.txt:4 (find_package):
    By not providing "Findcpptrace.cmake" in CMAKE_MODULE_PATH this project has
    asked CMake to find a package configuration file provided by "cpptrace",
    but CMake did not find one.

    Could not find a package configuration file provided by "cpptrace" with any
    of the following names:

      cpptraceConfig.cmake
      cpptrace-config.cmake

    ...
    -- Configuring incomplete, errors occurred!
    ```

    没有运行 `init_env_inside_docker.sh` 中 `step_cpptrace` 部分。

    以及其他相关错误，大部分情况都是因为没有运行 `init_env_inside_docker.sh` 中的安装部分。

??? failure "/usr/bin/install: cannot stat 'doc/jemalloc.html': No such file or directory"

    **make 编译时出错：**

    ```text
    /usr/bin/install: cannot stat 'doc/jemalloc.html': No such file or directory
    make[3]: *** [Makefile:518: install_doc_html] Error 1
    make[3]: *** Waiting for unfinished jobs....
    /usr/bin/install: cannot stat 'doc/jemalloc.3': No such file or directory
    make[3]: *** [Makefile:525: install_doc_man] Error 1
    ```

    在 `third_party/jemalloc` 运行：

    ```bash
    make && make install_bin install_include install_lib
    ```

    可以参考：[install: cannot stat ‘doc/jemalloc.html’: No such file or directory #231](https://github.com/jemalloc/jemalloc/issues/231)

??? failure "error: there are no arguments to ‘malloc_usable_size’ that depend on a template parameter, so a declaration of ‘malloc_usable_size’ must be available [-fpermissive]"
    **运行 `init_env_inside_docker.sh` 时出错：**

    ```text
    In file included from <RecStore Path>/third_party/folly/folly/IPAddress.cpp:28:
    <RecStore Path>/third_party/folly/folly/small_vector.h: In member function ‘auto folly::small_vector<T, M, P>::AllocationSize::operator()(void*) const’:
    <RecStore Path>/third_party/folly/folly/small_vector.h:1346:14: error: there are no arguments to ‘malloc_usable_size’ that depend on a template parameter, so a declaration of ‘malloc_usable_size’ must be available [-fpermissive]
    1346 |       return malloc_usable_size(ptr);
          |              ^~~~~~~~~~~~~~~~~~
    ```

    这个在新的版本已经修复了，如果仍然出现可以参考：
    
    * [build error: small_vector.h:642:47: error: ‘malloc_usable_size’ was not declared in this scope](https://github.com/facebook/hhvm/issues/4908)
    * [Build wangle encountered error: error: use of undeclared identifier 'malloc_usable_size' ](https://github.com/facebook/wangle/issues/65)
    * [MB-33900: Use folly Malloc.h for detecting malloc_usable_size](https://review.couchbase.com/c/platform/+/138406)

## Runtime Errors

??? failure "E20260117 01:49:19.761556 970021 engine_extendible_hash.h:88] shm malloc failed (OOM?), key: 93068 size: 512"
    通常是 `cache_ps.base_kv_config` 的容量预算过小导致。优先检查并调大以下字段之一：

    - `ENTRY_CAPACITY`
    - `DRAM_SIZE`
    - `SSD_SIZE`

    如果仍在使用旧配置字段，则对应为 `capacity`、`shmcapacity`、`ssdcapacity`。


??? failure "Fail to listen 127.0.0.1:xxxxx/xxxxx"
    说明端口已被旧进程占用，先检查并清理：

    ```bash
    ps aux | grep ps_server
    ```

??? failure "run_single_day.sh: ... bc: command not found"
    该问题只影响 `run_single_day.sh` 的耗时上报，不影响训练主体。可安装：

    ```bash
    sudo apt-get update && sudo apt-get install -y bc
    ```
