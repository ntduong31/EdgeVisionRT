#!/bin/bash
#
# EdgeVisionRT - Full Setup Script
# Installs all dependencies for Raspberry Pi 5 (Bookworm)
#

set -e

echo "=================================================="
echo "    EdgeVisionRT Setup Script for RPi 5"
echo "=================================================="

# Check for root/sudo
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or with sudo"
    echo "Usage: sudo ./setup.sh"
    exit 1
fi

echo ">> Updating system..."
apt update && apt upgrade -y

echo ">> Installing system dependencies..."
apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    libopencv-dev \
    libvulkan-dev \
    vulkan-tools \
    libomp-dev \
    glslang-dev \
    glslang-tools

echo ">> Configuring Framebuffer permissions..."
# Allow current user to access framebuffer
if [ ! -z "$SUDO_USER" ]; then
    usermod -aG video $SUDO_USER
    chmod 666 /dev/fb0
    echo "Added user $SUDO_USER to video group and set fb0 permissions."
else
    echo "Warning: script run as root without sudo, cannot detect user to add to video group."
fi

echo ">> Installing NCNN..."
# We use the local deps provided in the repo, but if we needed to build it:
# (This section is optional if we assume deps/ is populated, 
#  but good for a fresh environment setup if deps is missing)

NCNN_DIR="deps/ncnn-install"
if [ ! -d "$NCNN_DIR" ]; then
    echo "NCNN not found in deps/, building from source..."
    mkdir -p deps
    cd deps
    if [ ! -d "ncnn" ]; then
        git clone https://github.com/Tencent/ncnn.git
    fi
    cd ncnn
    mkdir -p build
    cd build
    cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake \
          -DNCNN_VULKAN=ON \
          -DNCNN_BUILD_EXAMPLES=OFF \
          -DNCNN_BUILD_TOOLS=OFF \
          -DNCNN_SYSTEM_GLSLANG=ON \
          ..
    make -j$(nproc)
    make install
    # Link to expected location
    cd ../..
    rm -rf ncnn-install
    mv ncnn/build/install ncnn-install
    cd ..
else
    echo "NCNN already installed in deps/ncnn-install."
fi

echo ">> Configuring CPU Governor service..."
# Create a systemd service to force performance mode on boot
cat > /etc/systemd/system/performance_cpu.service << EOF
[Unit]
Description=Set CPU governor to performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable performance_cpu.service
systemctl start performance_cpu.service
echo "CPU governor service enabled (performance mode)."

echo "=================================================="
echo "    Setup Complete!"
echo "=================================================="
echo "You can now build and run the project:"
echo "  ./build.sh"
echo "  ./run.sh"
