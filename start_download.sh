#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ttvla

# Kill old download attempts
kill $(ps aux | grep 'from_pretrained.*openvla' | grep -v grep | awk '{print $2}') 2>/dev/null
kill $(ps aux | grep 'snapshot_download' | grep -v grep | awk '{print $2}') 2>/dev/null

# Clean incomplete files
rm -f ~/.cache/huggingface/hub/models--openvla--openvla-7b/blobs/*.incomplete

# Run download script
export HF_ENDPOINT=https://hf-mirror.com
nohup python3 -u /home/liuyue/data/tt-vla/scripts/download_openvla.py \
  > /home/liuyue/data/tt-vla/logs/download_shards.log 2>&1 &

echo "Download started, PID=$!"
sleep 20
echo "--- Log after 20s ---"
cat /home/liuyue/data/tt-vla/logs/download_shards.log
