# Install FlagTree

## Option 1: Install from source

1. Clone FlagTree and enter the FlagTree directory.

    ```{code-block} bash
    #    Clone the FlagTree project from the remote Git repository to local
    git clone https://github.com/flagos-ai/flagtree
    #    Switch the current working directory to the cloned FlagTree project directory
    cd flagtree
    ```

2. Install Ubuntu and Python dependencies.

    ```{code-block} bash
    # Install dependencies for Ubuntu
    apt install zlib1g zlib1g-dev libxml2 libxml2-dev  # ubuntu
    # Install dependencies for Python
    # The dependencies are included in the requirements.txt in the flagtree/python directory.
    cd python; python3 -m pip install -r requirements.txt

    ```

3. Build and install FlagTree
    Below are the common commands used to build and install FlagTree. However, different backends have different requirements. For more information, see the "Install FlagTree for different backends" section.

    ```{code-block} bash
    # for branch main、triton_v3.2.x、triton_v3.3.x
    cd python
    # Set the environment variable to specify the computational backend
    export FLAGTREE_BACKEND=backendxxx
    # Install the package in development mode with verbose output and system dependencies
    python3 -m pip install . --no-build-isolation -v
    # Print the metadata and location of the installed flagtree package
    python3 -m pip show flagtree
    # Return to the home directory and print the file path of the installed Triton library
    cd; python3 -c 'import triton; print(triton.__path__)'
    ```

### Install FlagTree for different backends

This section includes how to install dependencies and FlagTree for different backends.

```{tip}
Automatic dependency library downloads may be limited by network conditions. You can manually download to the cache directory ~/.flagtree (modifiable via the FLAGTREE_CACHE_DIR environment variable). No need to manually set LLVM environment variables such as LLVM_BUILD_DIR.
```

Complete build commands for each backend:

#### ILUVATAR [iluvatar](https://github.com/flagos-ai/FlagTree/tree/main/third_party/iluvatar/)

Based on Triton 3.1, x64

1. Build and run environment

    Recommended: Use Ubuntu 20.04

2. Manually download the FlagTree dependencies

    ```{code} shell
    mkdir -p ~/.flagtree/iluvatar; cd ~/.flagtree/iluvatar
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatar-llvm18-x86_64_v0.4.0.tar.gz
    tar zxvf iluvatar-llvm18-x86_64_v0.4.0.tar.gz
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatarTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x86_64_v0.4.0.tar.gz
    tar zxvf iluvatarTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x86_64_v0.4.0.tar.gz

    ```

3. Manually download the Triton dependencies

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree
    # For Triton 3.1 (x64)
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.1.x-linux-x64.tar.gz
    sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.1.x-linux-x64.tar.gz
    ```

4. Command to build from source

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree/python
    export FLAGTREE_BACKEND=iluvatar
    python3 -m pip install . --no-build-isolation -v
    ```

#### KLX [xpu](https://github.com/flagos-ai/FlagTree/tree/main/third_party/xpu/)

Based on Triton 3.0, x64

1. Build and run environment

   - Recommended: Use the Docker image (22GB) [ubuntu_2004_x86_64_v30.tar](https://su.bcebos.com/klx-sdk-release-public/xpytorch/docker/ubuntu2004_v030/ubuntu_2004_x86_64_v30.tar)
   - Contact <kunlunxin-support@baidu.com> for support

2. Manually download the FlagTree dependencies

    ```{code} shell
    mkdir -p ~/.flagtree/xpu; cd ~/.flagtree/xpu
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/XTDK-llvm19-ubuntu2004_x86_64_v0.3.0.tar.gz
    tar zxvf XTDK-llvm19-ubuntu2004_x86_64_v0.3.0.tar.gz
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/xre-Linux-x86_64_v0.3.0.tar.gz
    tar zxvf xre-Linux-x86_64_v0.3.0.tar.gz
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/xpu-liblaunch_shared_so-ubuntu-x64_v0.3.1.tar.gz
    tar zxvf xpu-liblaunch_shared_so-ubuntu-x64_v0.3.1.tar.gz
    ```

3. Manually download the Triton dependencies

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree
    # For Triton 3.1 (x64)
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.1.x-linux-x64.tar.gz
    sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.1.x-linux-x64.tar.gz
    ```

4. Command to build from source

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree/python
    export FLAGTREE_BACKEND=xpu
    python3 -m pip install . --no-build-isolation -v
    ```

#### Moore Threads [mthreads](https://github.com/flagos-ai/FlagTree/tree/main/third_party/mthreads/)

Based on Triton 3.1, x64/aarch64

1. Build and run environment

    Recommended: Use [Dockerfile-ubuntu22.04-python3.10-mthreads](/dockerfiles/Dockerfile-ubuntu22.04-python3.10-mthreads)

2. Manually download the FlagTree dependencies

    ```{code} shell
    mkdir -p ~/.flagtree/mthreads; cd ~/.flagtree/mthreads
    # x64
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm19-glibc2.35-glibcxx3.4.30-x64_v0.4.0.tar.gz
    tar zxvf mthreads-llvm19-glibc2.35-glibcxx3.4.30-x64_v0.4.0.tar.gz
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x64_v0.4.1.tar.gz
    tar zxvf mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x64_v0.4.1.tar.gz
    # aarch64
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm19-glibc2.35-glibcxx3.4.30-aarch64_v0.4.0.tar.gz
    tar zxvf mthreads-llvm19-glibc2.35-glibcxx3.4.30-aarch64_v0.4.0.tar.gz
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-aarch64_v0.4.0.tar.gz
    tar zxvf mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-aarch64_v0.4.0.tar.gz
    ```

3. Manually download the Triton dependencies

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree
    # For Triton 3.1 (x64)
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.1.x-linux-x64.tar.gz
    sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.1.x-linux-x64.tar.gz
    ```

4. Command to build from source

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree/python
    export FLAGTREE_BACKEND=mthreads
    python3 -m pip install . --no-build-isolation -v
    ```

#### ARM China [aipu](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/aipu/)

Based on Triton 3.3, x64/arm64

1. Build and run environment

    Recommended: Use Ubuntu 22.04

2. Manually download the FlagTree dependencies

    llvm x64 in the simulated environment, llvm arm64 on the ARM development board

    ```{code} shell
    mkdir -p ~/.flagtree/aipu; cd ~/.flagtree/aipu
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-x64-clang16-lld16_v0.4.0.tar.gz
    tar zxvf llvm-a66376b0-ubuntu-x64-clang16-lld16_v0.4.0.tar.gz
    ```

3. Manually download the Triton dependencies

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree
    # For Triton 3.3 (x64)
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.3.x-linux-x64.tar.gz
    sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.3.x-linux-x64.tar.gz
    ```

4. Command to build from source

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree/python
    git checkout -b triton_v3.3.x origin/triton_v3.3.x
    export FLAGTREE_BACKEND=aipu
    python3 -m pip install . --no-build-isolation -v
    ```

#### Tsingmicro [tsingmicro](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/tsingmicro/)

Based on Triton 3.3, x64

1. Build and run environment

    Recommended: Use Ubuntu 20.04

2. Manually download the FlagTree dependencies

    ```{code} shell
    mkdir -p ~/.flagtree/tsingmicro; cd ~/.flagtree/tsingmicro
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.11-x64_v0.2.0.tar.gz
    tar zxvf tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.11-x64_v0.2.0.tar.gz
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tx8_depends_release_20250814_195126_v0.2.0.tar.gz
    tar zxvf tx8_depends_release_20250814_195126_v0.2.0.tar.gz
    ```

3. Manually download the Triton dependencies

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree
    # For Triton 3.3 (x64)
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.3.x-linux-x64.tar.gz
    sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.3.x-linux-x64.tar.gz
    ```

4. Command to build from source

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree/python
    git checkout -b triton_v3.3.x origin/triton_v3.3.x
    export TX8_DEPS_ROOT=~/.flagtree/tsingmicro/tx8_deps
    export FLAGTREE_BACKEND=tsingmicro
    python3 -m pip install . --no-build-isolation -v
    ```

#### Huawei Ascend [ascend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.2.x/third_party/ascend)

Based on Triton 3.2, aarch64

1. Build and run environment

   - Use docker file: [Dockerfile-ubuntu22.04-python3.11-ascend](/dockerfiles/Dockerfile-ubuntu22.04-python3.11-ascend) (Recommended)
   - Use docker image <https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/docker_image_cann-8.2.rc1.alpha003-a3-ubuntu22.04-py3.11-flagtree.tar.gz> (5.4GB)

2. Reinstall CANN Toolkit and OPS
   1. Register an account at <https://www.hiascend.com/developer/download/community/result?module=cann>, and download the CANN Toolkit and OPS.
   2. Install the CANN Tookit and OPS.

    ```{code} shell
    # 1. Install CANN Toolkit (Required for all)
    chmod +x Ascend-cann-toolkit_8.5.0_linux-aarch64.run

    # 2. Install CANN OPS (Select ONE option based on your hardware)
    ./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install
    # Option A: cann-ops for 910B (A2)
    chmod +x Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
    # Option B: cann-ops for 910C (A3)
    chmod +x Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
    ```

3. Manually download the FlagTree dependencies

    ```{code} shell
    mkdir -p ~/.flagtree/ascend; cd ~/.flagtree/ascend
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz
    tar zxvf llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz
    ```

4. Manually download the Triton dependencies

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree
    # For Triton 3.2 (aarch64)
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.2.x-linux-aarch64.tar.gz
    sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.2.x-linux-aarch64.tar.gz
    ```

5. Command to build from source

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree/python
    git checkout -b triton_v3.2.x origin/triton_v3.2.x
    export FLAGTREE_BACKEND=ascend
    python3 -m pip install . --no-build-isolation -v
    ```

#### HYGON [hcu](https://github.com/flagos-ai/FlagTree/tree/main/third_party/hcu/)

  Based on Triton 3.0, x64

1. Build and run environment

    Recommended: Use [Dockerfile-ubuntu22.04-python3.10-hcu](/dockerfiles/Dockerfile-ubuntu22.04-python3.10-hcu)

2. Manually download the FlagTree dependencies

    ```{code} shell
    mkdir -p ~/.flagtree/hcu; cd ~/.flagtree/hcu
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz
    tar zxvf hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz
    ```

3. Manually download the Triton dependencies

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree
    # For Triton 3.1 (x64)
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.1.x-linux-x64.tar.gz
    sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.1.x-linux-x64.tar.gz
    ```

4. Command to build from source

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree/python
    export FLAGTREE_BACKEND=hcu
    python3 -m pip install . --no-build-isolation -v
    ```

#### Enflame [enflame](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/enflame/)

Based on Triton 3.3, x64

1. Build and run environment

    Recommended: Use the Docker image (2.4GB) <https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-flagtree-0.3.1.tar.gz>

2. Manually download the FlagTree dependencies

    ```{code} shell
    mkdir -p ~/.flagtree/enflame; cd ~/.flagtree/enflame
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
    tar zxvf enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
    ```

3. Manually download the Triton dependencies

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree
    # For Triton 3.3 (x64)
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.3.x-linux-x64.tar.gz
    sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.3.x-linux-x64.tar.gz
    ```

4. Command to build from source

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree/python
    export FLAGTREE_BACKEND=enflame
    python3 -m pip install . --no-build-isolation -v
    ```

#### Sunrise [sunrise](https://github.com/flagos-ai/FlagTree/tree/triton_v3.4.x/third_party/sunrise/)

Based on Triton 3.4, x64

1. Build and run environment

    Recommended: Use Ubuntu 22.04

2. Manually download the FlagTree dependencies

    ```{code} shell
    mkdir -p ~/.flagtree/sunrise; cd ~/.flagtree/sunrise
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/sunrise-llvm21-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
    tar zxvf sunrise-llvm21-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/sunriseTritonPlugin-cpython3.10-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
    tar zxvf sunriseTritonPlugin-cpython3.10-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
    ```

3. Manually download the Triton dependencies

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree
    # For Triton 3.4 (x64)
    wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.4.x-linux-x64.tar.gz
    sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.4.x-linux-x64.tar.gz
    ```

4. Command to build from source

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree
    export TRITON_BUILD_WITH_CLANG_LLD=1
    export TRITON_OFFLINE_BUILD=1
    export TRITON_BUILD_PROTON=OFF
    export FLAGTREE_BACKEND=sunrise
    python3 -m pip install . --no-build-isolation -v
    ```

#### NVIDIA & AMD [nvidia](/third_party/nvidia/) & [amd](/third_party/amd/)

Based on Triton 3.1/3.2/3.3/3.4/3.5, x64/arm64

1. Build and run environment

    Recommended: Use the Docker image (12GB) <https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/docker_image_nvidia_pytorch_25.05-py3.tar.gz>

2. Manually download the LLVM

    ```{code} shell
    cd ${YOUR_LLVM_DOWNLOAD_DIR}
    # For Triton 3.1
    wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-10dc3a8e-ubuntu-x64.tar.gz
    tar zxvf llvm-10dc3a8e-ubuntu-x64.tar.gz
    export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-10dc3a8e-ubuntu-x64
    # For Triton 3.2
    wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-86b69c31-ubuntu-x64.tar.gz
    tar zxvf llvm-86b69c31-ubuntu-x64.tar.gz
    export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-86b69c31-ubuntu-x64
    # For Triton 3.3
    wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-a66376b0-ubuntu-x64.tar.gz
    tar zxvf llvm-a66376b0-ubuntu-x64.tar.gz
    export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-a66376b0-ubuntu-x64
    # For Triton 3.4
    wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-8957e64a-ubuntu-x64.tar.gz
    tar zxvf llvm-8957e64a-ubuntu-x64.tar.gz
    export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-8957e64a-ubuntu-x64
    # For Triton 3.5
    wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-7d5de303-ubuntu-x64.tar.gz
    tar zxvf llvm-7d5de303-ubuntu-x64.tar.gz
    export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-7d5de303-ubuntu-x64
    #
    export LLVM_INCLUDE_DIRS=$LLVM_SYSPATH/include
    export LLVM_LIBRARY_DIR=$LLVM_SYSPATH/lib
    ```

3. Manually download the Triton dependencies

    Refer to [Offline build support: pre-downloading dependency packages](/documents/build.md#offline-build-support).

4. Command to build from source

    ```{code} shell
    cd ${YOUR_CODE_DIR}/FlagTree
    cd python  # For Triton 3.1, 3.2, 3.3, you need to enter the python directory to build
    git checkout main                                   # For Triton 3.1
    git checkout -b triton_v3.2.x origin/triton_v3.2.x  # For Triton 3.2
    git checkout -b triton_v3.3.x origin/triton_v3.3.x  # For Triton 3.3
    git checkout -b triton_v3.4.x origin/triton_v3.4.x  # For Triton 3.4
    git checkout -b triton_v3.5.x origin/triton_v3.5.x  # For Triton 3.5
    unset FLAGTREE_BACKEND
    python3 -m pip install . --no-build-isolation -v
    # If you need to build other backends afterward, you should clear LLVM-related environment variables
    unset LLVM_SYSPATH LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR
    ```

### Offline build support

The above introduced how dependencies can be manually downloaded for various FlagTree backends during build time to avoid network environment limitations. Since Triton builds originally come with some dependency packages, we provide pre-downloaded packages that can be manually installed in your environment to prevent getting stuck at the automatic download stage during the build process.

```{code} shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.1 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.1.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.1.x-linux-x64.tar.gz
# For Triton 3.2 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.2.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.2.x-linux-x64.tar.gz
# For Triton 3.2 (aarch64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.2.x-linux-aarch64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.2.x-linux-aarch64.tar.gz
# For Triton 3.3 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.3.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.3.x-linux-x64.tar.gz
# For Triton 3.4 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.4.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.4.x-linux-x64.tar.gz
# For Triton 3.5 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.5.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.5.x-linux-x64.tar.gz
```

After executing the above script, the original ~/.triton directory will be renamed, and a new ~/.triton directory will be created to store the pre-downloaded packages.

### Q&A

Q: After installation, running the program reports: version GLIBC or GLIBCXX not found

A: Check which GLIBC / GLIBCXX versions are supported by libc.so.6 and libstdc++.so.6.0.30 in your environment:

```{code} shell
strings /lib/x86_64-linux-gnu/libc.so.6 |grep GLIBC
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 | grep GLIBCXX
```

If the required GLIBC / GLIBCXX version is supported, you can also try:

```{code} shell
export LD_PRELOAD="/lib/x86_64-linux-gnu/libc.so.6"  # If GLIBC cannot be found
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30"  # If GLIBCXX cannot be found
export LD_PRELOAD="/lib/x86_64-linux-gnu/libc.so.6 \
    /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30"  # If neither GLIBC nor GLIBCXX can be found
```

## Option 2: Install FlagTree wheel package

If you do not wish to build from source, you can directly pull and install whl (partial backend support).
The best practice to avoid environment compatibility issues is to use the image recommended in [Tips for building](/documents/build.md#tips-for-building).

1. Install PyTorch
2. Uninstall Triton

    ```{code} python
    python3 -m pip uninstall -y triton  # TODO: automatically uninstall triton
    RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
        --trusted-host=https://resource.flagos.net"
    ```

3. Install FlagTree and Triton

    |Backend   |Install command<br>(The version corresponds to the git tag)|Triton<br>version|Python<br>version|libc.so &<br>libstdc++.so<br>version|
    |:---------|:---------|:---------|:---------|:---------|
    |nvidia    |python3 -m pip install flagtree==0.4.0 $RES              |3.1|3.10<br>3.11<br>3.12|GLIBC_2.30<br>GLIBCXX_3.4.28<br>CXXABI_1.3.12|
    |nvidia    |python3 -m pip install flagtree==0.4.0+3.2 $RES          |3.2|3.10<br>3.11<br>3.12|GLIBC_2.30<br>GLIBCXX_3.4.28<br>CXXABI_1.3.12|
    |nvidia    |python3 -m pip install flagtree==0.4.0+3.3 $RES          |3.3|3.10<br>3.11<br>3.12|GLIBC_2.30<br>GLIBCXX_3.4.28<br>CXXABI_1.3.12|
    |nvidia    |python3 -m pip install flagtree==0.4.1+3.5 $RES          |3.5|3.12|GLIBC_2.39<br>GLIBCXX_3.4.33<br>CXXABI_1.3.15|
    |iluvatar  |python3 -m pip install flagtree==0.4.0+iluvatar3.1 $RES  |3.1|3.10|GLIBC_2.35<br>GLIBCXX_3.4.30<br>CXXABI_1.3.13|
    |mthreads  |python3 -m pip install flagtree==0.4.0+mthreads3.1 $RES  |3.1|3.10|GLIBC_2.35<br>GLIBCXX_3.4.30<br>CXXABI_1.3.13|
    |metax     |python3 -m pip install flagtree==0.4.0rc1+metax3.1 $RES  |3.1|3.10|GLIBC_2.39<br>GLIBCXX_3.4.33<br>CXXABI_1.3.15|
    |ascend    |python3 -m pip install flagtree==0.4.1+ascend3.2 $RES    |3.2|3.11|GLIBC_2.34<br>GLIBCXX_3.4.24<br>CXXABI_1.3.11|
    |tsingmicro|python3 -m pip install flagtree==0.4.0+tsingmicro3.3 $RES|3.3|3.10|GLIBC_2.30<br>GLIBCXX_3.4.28<br>CXXABI_1.3.12|
    |hcu       |python3 -m pip install flagtree==0.4.0+hcu3.0 $RES       |3.0|3.10|GLIBC_2.35<br>GLIBCXX_3.4.30<br>CXXABI_1.3.13|
    |enflame   |python3 -m pip install flagtree==0.4.0+enflame3.3 $RES   |3.3|3.10|GLIBC_2.35<br>GLIBCXX_3.4.30<br>CXXABI_1.3.13|
    |sunrise   |python3 -m pip install flagtree==0.4.0+sunrise3.4 $RES   |3.4|3.10|GLIBC_2.39<br>GLIBCXX_3.4.33<br>CXXABI_1.3.15|

## Running tests

After installation, you can generally run the following tests. For specific backend support tests, please refer to .github/workflow/${backend_name}-build-and-test.yml in the corresponding branch.

```{code} shell
# nvidia/amd
cd python/test/unit
python3 -m pytest -s
# other backends
cd third_party/${backend_name}/python/test/unit
python3 -m pytest -s
```
