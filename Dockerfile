# ============================================================
# spr_vision_26 Docker 镜像
# 用途: 编译环境 / CI/CD / 离线测试 / Demo 运行
# 目标架构: linux/amd64（Intel NUC 部署环境）
# Apple Silicon Mac 上会用 QEMU 模拟，编译较慢但结果一致
# ============================================================

# 默认 linux/amd64（Intel NUC 部署环境）
# Apple Silicon Mac 上会用 QEMU 模拟，编译较慢但结果一致
ARG PLATFORM=linux/amd64
FROM --platform=${PLATFORM} ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

# ---- 换国内镜像源（可选，加速 apt；若失败可注释此行） ----
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# ============================================================
# 1. 系统工具链 + 核心依赖 (apt)
# ============================================================
RUN apt update && apt install -y --no-install-recommends \
    # --- 编译工具链 ---
    g++ \
    cmake \
    make \
    git \
    wget \
    ca-certificates \
    gnupg \
    # --- 核心库 ---
    libopencv-dev \
    libfmt-dev \
    libeigen3-dev \
    libspdlog-dev \
    libyaml-cpp-dev \
    libusb-1.0-0-dev \
    nlohmann-json3-dev \
    # --- Ceres 依赖 ---
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    # --- 运行时工具 ---
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 2. OpenVINO 2024.6.0 (C++ Runtime)
# ============================================================
# Intel apt 源
RUN wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
    gpg --dearmor -o /usr/share/keyrings/intel-sw-repositories.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/intel-sw-repositories.gpg] https://apt.repos.intel.com/openvino/2024 ubuntu22 main" \
    > /etc/apt/sources.list.d/intel-openvino-2024.list && \
    apt update && apt install -y --no-install-recommends \
    openvino-2024.6.0 \
    && rm -rf /var/lib/apt/lists/*

# apt 安装 openvino-2024.6.0 后路径就是 /opt/intel/openvino_2024.6.0
ENV OpenVINO_DIR=/opt/intel/openvino_2024.6.0/runtime/cmake
ENV LD_LIBRARY_PATH=/opt/intel/openvino_2024.6.0/runtime/lib/intel64:/usr/local/lib

# ============================================================
# 3. Ceres Solver (源码编译)
# ============================================================
RUN git clone --depth 1 --branch 2.2.0 \
    https://github.com/ceres-solver/ceres-solver.git /tmp/ceres && \
    cd /tmp/ceres && \
    cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_BENCHMARKS=OFF && \
    cmake --build build -j$(nproc) && \
    cmake --install build && \
    rm -rf /tmp/ceres

# ============================================================
# 4. 项目源码 & 编译
# ============================================================
WORKDIR /workspace/spr_vision

COPY . .

# 编译（Release 模式更快，适合 CI；Debug 也可按需切换）
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc)

# ============================================================
# 5. 入口
# ============================================================
# 此镜像仅用于编译 / CI，Apple Silicon 不支持 OpenVINO 推理
# 用法:
#   docker run --rm -it spr-vision bash          # 交互式进入
#   docker run --rm spr-vision cmake --build build -j$(nproc)  # 手动编译
CMD ["bash"]