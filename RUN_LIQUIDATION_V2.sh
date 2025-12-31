#!/bin/bash

# 🌊 Moon Dev Liquidation Agent V2 - Execution Script
# Handles PYTHONPATH setup for HPC/Apptainer environments

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

echo -e "${GREEN}"
echo "=========================================="
echo "🌊 MOON DEV LIQUIDATION AGENT V2"
echo "=========================================="
echo -e "${NC}"

# Determine script directory (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${YELLOW}[INFO]${NC} Project root: $SCRIPT_DIR"

# Verify project structure
if [ ! -d "$SCRIPT_DIR/src/agents" ]; then
    echo -e "${RED}[ERROR]${NC} Cannot find src/agents directory"
    echo -e "${RED}[ERROR]${NC} Current directory: $(pwd)"
    echo -e "${RED}[ERROR]${NC} Script directory: $SCRIPT_DIR"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/src/agents/liquidation_agent_v2.py" ]; then
    echo -e "${RED}[ERROR]${NC} Cannot find liquidation_agent_v2.py"
    exit 1
fi

echo -e "${GREEN}[OK]${NC} Found project structure"

# Set PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/src:${PYTHONPATH}"

echo -e "${YELLOW}[INFO]${NC} PYTHONPATH set to:"
echo "  ${SCRIPT_DIR}"
echo "  ${SCRIPT_DIR}/src"

# Check Python installation
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}[ERROR]${NC} Python not found"
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

echo -e "${GREEN}[OK]${NC} Using: $($PYTHON_CMD --version)"

# Change to project root
cd "$SCRIPT_DIR"
echo -e "${YELLOW}[INFO]${NC} Working directory: $(pwd)"

# Run the agent
echo -e "${GREEN}"
echo "========================================"
echo "🚀 Starting agent..."
echo "========================================"
echo -e "${NC}"

$PYTHON_CMD src/agents/liquidation_agent_v2.py "$@"

exit_code=$?

echo -e "${YELLOW}[INFO]${NC} Agent exited with code: $exit_code"
exit $exit_code
