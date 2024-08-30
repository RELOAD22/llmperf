#!/bin/bash

# Check if the required arguments are provided
if [ $# -ne 4 ]; then
  echo "Usage: $0 <backend_name> <port> <model_name> <model_dir>"
  exit 1
fi

# Assign the input arguments to variables
backend_name=$1
port=$2
model_name=$3
model_dir=$4

# Run the benchmarks
echo "Running benchmarks for backend: $backend_name"
echo "Using port: $port"
echo "Using model: $model_name"
echo "Using model directory: $model_dir"

# Testing Configurations List for $concurrent_requests $isl $osl
config_list=(
  "concurrent_requests=1 isl=128 osl=128"
  "concurrent_requests=1 isl=2048 osl=128"
  "concurrent_requests=1 isl=128 osl=2048"
  "concurrent_requests=1 isl=2048 osl=2048"
  "concurrent_requests=128 isl=128 osl=2048"
  "concurrent_requests=256 isl=128 osl=2048"
)

for config in "${config_list[@]}"; do
  eval $config
  echo "Running benchmark for configuration: $concurrent_requests $isl $osl"
  python -m sglang.bench_serving --backend $backend_name --num-prompt $concurrent_requests --host 127.0.0.1 --port $port --tokenizer $model_dir --random-input $isl --random-output $osl --random-range-ratio 1 --dataset-name random
done

echo "Benchmarks completed"