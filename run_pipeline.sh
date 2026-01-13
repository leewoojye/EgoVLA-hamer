#!/bin/bash

# 사용 예시:
# ./run_pipeline.sh <비디오 경로> <출력 폴더 기본 이름> <언어 지시사항>

# 기본값 설정
VIDEO_PATH=${1:-"data_in/egovla_demo/공꺼내기_화장실_3.MOV"}
OUTPUT_ROOT=${2:-"data_out"}
INSTRUCTION=${3:-"Get the ball out"}

# Squeeze the towel
# Grab the spoon
# Get the ball out

# 가상 환경 활성화 (필요한 경우)
# source activate hamer 
# 또는
# conda activate hamer

echo "Running preprocess pipeline..."
echo "Video: $VIDEO_PATH"
echo "Output Root Base: $OUTPUT_ROOT"
echo "Instruction: $INSTRUCTION"

python preprocess_pipeline.py "$VIDEO_PATH" "$OUTPUT_ROOT" "$INSTRUCTION"
