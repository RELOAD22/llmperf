#!/bin/bash

# Check if the required arguments are provided
if [ $# -ne 5 ]; then
  echo "Usage: $0 <backend_name> <port> <model_name> <model_dir> <results_dir>"
  exit 1
fi

# Assign the input arguments to variables
backend_name=$1
port=$2
model_name=$3
model_dir=$4
results_dir=$5

# Run the benchmarks
echo "Running benchmarks for backend: $backend_name"
echo "Using port: $port"
echo "Using model: $model_name"
echo "Using model directory: $model_dir"
results_dir=$results_dir/$backend_name
mkdir -p $results_dir
echo "Saving results to: $results_dir"

# Add your benchmark commands here
export OPENAI_API_KEY=EMPTY; export OPENAI_API_BASE="http://127.0.0.1:$port/v1/"

# Testing Configurations List for $concurrent_requests $total_requests $isl $osl
config_list=(
  "concurrent_requests=1 total_requests=5 isl=128 osl=128"
  "concurrent_requests=1 total_requests=5 isl=2048 osl=128"
  "concurrent_requests=1 total_requests=5 isl=128 osl=2048"
  "concurrent_requests=1 total_requests=5 isl=2048 osl=2048"
  "concurrent_requests=4 total_requests=12 isl=128 osl=2048"
  "concurrent_requests=8 total_requests=24 isl=128 osl=2048"
  "concurrent_requests=16 total_requests=48 isl=128 osl=2048"
  "concurrent_requests=32 total_requests=32 isl=128 osl=2048"
  "concurrent_requests=64 total_requests=64 isl=128 osl=2048"
  "concurrent_requests=128 total_requests=128 isl=128 osl=2048"
  "concurrent_requests=256 total_requests=256 isl=128 osl=2048"
)

for config in "${config_list[@]}"; do
  eval $config
  echo "Running benchmark for configuration: $concurrent_requests $total_requests $isl $osl"
  python token_benchmark_ray.py --model $model_name --mean-input-tokens $isl --stddev-input-tokens 0 --mean-output-tokens $osl --stddev-output-tokens 0 --max-num-completed-requests $total_requests --timeout 600 --num-concurrent-requests $concurrent_requests --results-dir $results_dir --llm-api openai --additional-sampling-params '{}' --model-dir $model_dir
done

echo "Benchmarks completed"