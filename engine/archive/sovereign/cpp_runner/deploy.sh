#!/bin/bash
# Deploy and build C++ runner on VPS

VPS="root@31.97.211.217"
REMOTE_DIR="/root/sovereign/cpp_runner"

echo "Deploying C++ Blockchain Runner to VPS..."

# Create remote directory
ssh $VPS "mkdir -p $REMOTE_DIR/src $REMOTE_DIR/include"

# Copy files
scp CMakeLists.txt $VPS:$REMOTE_DIR/
scp build.sh $VPS:$REMOTE_DIR/
scp include/*.hpp $VPS:$REMOTE_DIR/include/
scp src/*.cpp $VPS:$REMOTE_DIR/src/

# Build on VPS
ssh $VPS "cd $REMOTE_DIR && chmod +x build.sh && ./build.sh"
