# Find the GPU with least VRAM used

# Get GPU information using nvidia-smi
gpu_info=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits)

# Parse GPU information and find the GPU with the least VRAM usage
min_vram_gpu=""
min_vram_usage=999999999999 # Set an initial high value

while IFS=, read -r gpu_index total_vram used_vram; do
    # Calculate VRAM usage
    vram_usage=$((used_vram))

    # Check if the current GPU has lower VRAM usage
    if [ $vram_usage -lt $min_vram_usage ]; then
        min_vram_usage=$vram_usage
        min_vram_gpu=$gpu_index
    fi
done <<< "$gpu_info"

# Print the GPU with the least VRAM usage
echo "$min_vram_gpu"
