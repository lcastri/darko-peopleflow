#!/bin/bash
# This script starts the container in detached (background) mode.

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo " "
echo "Checking host graphics configuration..."

# --- Grant Docker access to the X Server ---
if [ -n "$DISPLAY" ]; then
    xhost +local:docker > /dev/null 2>&1
    echo -e "${GREEN}✔ X11 permissions granted to local Docker containers.${NC}"
else
    echo -e "${RED}WARNING: No DISPLAY detected. GUI applications may not work.${NC}"
fi

# --- Get host renderer ---
host_renderer=$(glxinfo -B | grep "OpenGL renderer string" | sed 's/OpenGL renderer string: //')

if [ -z "$host_renderer" ]; then
    echo -e "${RED}ERROR: Could not detect OpenGL renderer on host.${NC}"
    exit 1
fi

echo -e "${GREEN}✔ Host renderer detected:${NC} ${host_renderer}"

# Check for known bad renderers
STRICT_GPU_CHECK=${STRICT_GPU_CHECK:-0}

if echo "$host_renderer" | grep -q -E "llvmpipe|Mesa Intel"; then
    if [ "$STRICT_GPU_CHECK" -eq 1 ]; then
        echo -e "${RED}ERROR: Host is using CPU rendering.${NC}"
        exit 1
    else
        echo -e "${YELLOW}WARNING: Host is using CPU rendering. Performance will be slow.${NC}"
    fi
fi
echo -e "${GREEN}✔ GPU renderer detected. Proceeding to launch Docker...${NC}"

echo " "
echo "Starting the container in the background..."
docker compose up -d

echo " "
echo -e "${GREEN}✔ Container is running.${NC}"
echo -e "${YELLOW}You can now run './docshell.sh' to open a shell inside the container.${NC}"
echo -e "${YELLOW}You can now run './docstop.sh' to stop the container.${NC}"