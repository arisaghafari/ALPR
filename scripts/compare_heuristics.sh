#!/bin/bash
# =============================================================================
# Heuristics Comparison Script
# Runs the ALPR pipeline with and without heuristics to compare performance
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

INPUT_VIDEO="${1:-/opt/nvidia/deepstream/deepstream-7.1/sources/alpr_project/input_videos/sample.mp4}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/output/comparison}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}    HEURISTICS COMPARISON TEST${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""
echo -e "Script dir:  ${YELLOW}$SCRIPT_DIR${NC}"
echo -e "Input video: ${YELLOW}$INPUT_VIDEO${NC}"
echo -e "Output dir:  ${YELLOW}$OUTPUT_DIR${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Test 1: WITH Heuristics (default)
# =============================================================================
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}TEST 1: WITH Heuristics${NC}"
echo -e "${GREEN}=============================================${NC}"

python3 "$SCRIPT_DIR/alpr_deepstream_python.py" \
    -i "$INPUT_VIDEO" \
    -o "$OUTPUT_DIR/output_WITH_heuristics.mp4" \
    2>&1 | tee "$OUTPUT_DIR/log_WITH_heuristics.txt"

echo ""
echo -e "${YELLOW}Results saved to: $OUTPUT_DIR/log_WITH_heuristics.txt${NC}"
echo ""

# Wait for GPU memory to be fully released
echo -e "${YELLOW}Waiting 10 seconds for GPU cleanup...${NC}"
sleep 10

# =============================================================================
# Test 2: WITHOUT Heuristics
# =============================================================================
echo -e "${RED}=============================================${NC}"
echo -e "${RED}TEST 2: WITHOUT Heuristics${NC}"
echo -e "${RED}=============================================${NC}"

python3 "$SCRIPT_DIR/alpr_deepstream_python.py" \
    -i "$INPUT_VIDEO" \
    -o "$OUTPUT_DIR/output_WITHOUT_heuristics.mp4" \
    --no-heuristics \
    2>&1 | tee "$OUTPUT_DIR/log_WITHOUT_heuristics.txt"

echo ""
echo -e "${YELLOW}Results saved to: $OUTPUT_DIR/log_WITHOUT_heuristics.txt${NC}"
echo ""

# =============================================================================
# Compare Results
# =============================================================================
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}COMPARISON SUMMARY${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

echo -e "${GREEN}--- WITH Heuristics ---${NC}"
grep -E "Processing time|Average FPS|Stable plates|Total unique|High-density activations|Total vehicles filtered" "$OUTPUT_DIR/log_WITH_heuristics.txt" | tail -10

echo ""
echo -e "${RED}--- WITHOUT Heuristics ---${NC}"
grep -E "Processing time|Average FPS|Stable plates|Total unique|High-density activations|Total vehicles filtered|DISABLED" "$OUTPUT_DIR/log_WITHOUT_heuristics.txt" | tail -10

echo ""
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}Output files:${NC}"
echo -e "  Videos:"
echo -e "    WITH:    $OUTPUT_DIR/output_WITH_heuristics.mp4"
echo -e "    WITHOUT: $OUTPUT_DIR/output_WITHOUT_heuristics.mp4"
echo -e "  Logs:"
echo -e "    WITH:    $OUTPUT_DIR/log_WITH_heuristics.txt"
echo -e "    WITHOUT: $OUTPUT_DIR/log_WITHOUT_heuristics.txt"
echo -e "${BLUE}=============================================${NC}"
