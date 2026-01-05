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

echo ">> Fixing runtime directory permissions..."
# Fix Qt/OpenCV runtime directory permissions issue
if [ ! -z "$SUDO_USER" ]; then
    SUDO_UID=$(id -u $SUDO_USER)
    RUNTIME_DIR="/run/user/$SUDO_UID"
    if [ -d "$RUNTIME_DIR" ]; then
        chmod 0700 "$RUNTIME_DIR"
        echo "Fixed permissions for $RUNTIME_DIR to 0700"
    fi
fi

echo ">> Installing NCNN with Vulkan and INT8 support..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCNN_VULKAN_DIR="$SCRIPT_DIR/deps/ncnn-vulkan-install"

# Check if NCNN is properly installed with cmake config files
if [ -f "$NCNN_VULKAN_DIR/lib/cmake/ncnn/ncnnConfig.cmake" ]; then
    echo "NCNN already properly installed in deps/ncnn-vulkan-install with cmake config."
else
    echo "Building NCNN from source with Vulkan + INT8 support..."
    
    # Clean up any incomplete installations
    cd "$SCRIPT_DIR"
    rm -rf deps/ncnn deps/ncnn-install deps/ncnn-vulkan-install
    mkdir -p deps
    cd deps
    
    # Clone NCNN repository
    echo "Cloning NCNN repository..."
    git clone --depth=1 https://github.com/Tencent/ncnn.git
    
    # Build NCNN with Vulkan support
    echo "Configuring NCNN build..."
    cd ncnn
    mkdir -p build-vulkan
    cd build-vulkan
    
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DNCNN_VULKAN=ON \
        -DNCNN_BUILD_EXAMPLES=OFF \
        -DNCNN_BUILD_TOOLS=ON \
        -DNCNN_SYSTEM_GLSLANG=ON \
        -DNCNN_BUILD_BENCHMARK=OFF \
        -DNCNN_INT8=ON \
        -DCMAKE_INSTALL_PREFIX="$NCNN_VULKAN_DIR" \
        ..
    
    if [ $? -ne 0 ]; then
        echo "Error: CMake configuration failed!"
        exit 1
    fi
    
    # Build NCNN
    echo "Building NCNN (this may take 5-10 minutes)..."
    make -j$(nproc)
    
    if [ $? -ne 0 ]; then
        echo "Error: NCNN build failed!"
        exit 1
    fi
    
    # Install NCNN
    echo "Installing NCNN..."
    make install
    
    if [ $? -ne 0 ]; then
        echo "Error: NCNN installation failed!"
        exit 1
    fi
    
    # Verify installation
    if [ ! -f "$NCNN_VULKAN_DIR/lib/cmake/ncnn/ncnnConfig.cmake" ]; then
        echo "Error: NCNN cmake config files not found after installation!"
        exit 1
    fi
    
    echo "NCNN successfully installed with Vulkan + INT8 support."
    cd "$SCRIPT_DIR"
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
