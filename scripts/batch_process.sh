#!/bin/bash
# Batch process multiple video files with ALPR pipeline
# Usage: ./batch_process.sh [input_dir] [output_dir]
# Default: processes all videos in input_videos/ and saves to output_videos/

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/alpr_deepstream_python.py"

# Default directories (your project structure)
DEFAULT_INPUT_DIR="/opt/nvidia/deepstream/deepstream-7.1/sources/alpr_project/input_videos"
DEFAULT_OUTPUT_DIR="/opt/nvidia/deepstream/deepstream-7.1/sources/alpr_project/output_videos"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "[ERROR] Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Function to process a single video
process_video() {
    local input_file="$1"
    local output_dir="$2"
    
    # Get filename without path and extension
    local basename=$(basename "$input_file")
    local name="${basename%.*}"
    local ext="${basename##*.}"
    
    # Create output filename
    local output_file="${output_dir}/${name}_output.mp4"
    
    echo ""
    echo "=========================================="
    echo "Processing: $basename"
    echo "Output:     $output_file"
    echo "=========================================="
    
    # Run the ALPR pipeline
    python3 "$PYTHON_SCRIPT" -i "$input_file" -o "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Completed: $basename -> $(basename "$output_file")"
    else
        echo "[FAILED] Error processing: $basename"
    fi
}

# Main logic
if [ $# -lt 1 ]; then
    # No arguments - use default directories
    INPUT_DIR="$DEFAULT_INPUT_DIR"
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
    echo "[BATCH] Using default directories:"
elif [ -d "$1" ]; then
    # First argument is a directory
    INPUT_DIR="$1"
    OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"
else
    # Process individual files (handled below)
    INPUT_DIR=""
fi

# Process directory mode
if [ -n "$INPUT_DIR" ] && [ -d "$INPUT_DIR" ]; then
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    echo "[BATCH] Input directory:  $INPUT_DIR"
    echo "[BATCH] Output directory: $OUTPUT_DIR"
    
    # Find all video files using find command (more compatible)
    VIDEO_FILES=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "video*.mp4" \) | sort)
    VIDEO_COUNT=$(echo "$VIDEO_FILES" | grep -c .)
    
    if [ "$VIDEO_COUNT" -eq 0 ]; then
        echo "[BATCH] No video files found in $INPUT_DIR"
        exit 1
    fi
    
    echo "[BATCH] Found $VIDEO_COUNT video files"
    echo ""
    
    # Process each video
    PROCESSED=0
    while IFS= read -r video; do
        if [ -f "$video" ]; then
            PROCESSED=$((PROCESSED + 1))
            echo "[BATCH] Processing $PROCESSED/$VIDEO_COUNT"
            process_video "$video" "$OUTPUT_DIR"
        fi
    done <<< "$VIDEO_FILES"
    
else
    # Process individual files passed as arguments
    OUTPUT_DIR="./output"
    FILES=()
    
    # Parse arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            *)
                if [ -f "$1" ]; then
                    FILES+=("$1")
                else
                    echo "[WARNING] File not found: $1"
                fi
                shift
                ;;
        esac
    done
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    VIDEO_COUNT=${#FILES[@]}
    echo "[BATCH] Processing $VIDEO_COUNT video files"
    echo "[BATCH] Output directory: $OUTPUT_DIR"
    echo ""
    
    PROCESSED=0
    for video in "${FILES[@]}"; do
        ((PROCESSED++))
        echo "[BATCH] Processing $PROCESSED/$VIDEO_COUNT"
        process_video "$video" "$OUTPUT_DIR"
    done
fi

echo ""
echo "=========================================="
echo "[BATCH] All processing complete!"
echo "=========================================="
