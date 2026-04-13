#!/bin/bash
# Remap absolute paths in training JSON to match remote server
# Usage: bash scripts/remap_paths.sh /path/to/data_root
set -e

DATA_ROOT="${1:?Usage: bash scripts/remap_paths.sh /path/to/data_root}"
cd "$(dirname "$0")/.."

OLD_FRAMES="/home/user2/zouyueying/ViF-CoT-4K/parsed_frames"
NEW_FRAMES="${DATA_ROOT}/parsed_frames"

for JSON_FILE in train/data/ladm_physcot.json train/data/ladm_sft_local.json; do
    if [ ! -f "$JSON_FILE" ]; then
        echo "[SKIP] $JSON_FILE not found"
        continue
    fi
    echo "Remapping: $JSON_FILE"
    cp "$JSON_FILE" "${JSON_FILE}.bak"
    python3 -c "
import json
with open('${JSON_FILE}') as f:
    data = json.load(f)
for s in data:
    if 'images' in s:
        s['images'] = [p.replace('${OLD_FRAMES}', '${NEW_FRAMES}') for p in s['images']]
with open('${JSON_FILE}', 'w') as f:
    json.dump(data, f, ensure_ascii=False)
print(f'  Done: {len(data)} samples')
"
done

# Verify
python3 -c "
import json, os
data = json.load(open('train/data/ladm_physcot.json'))
img = data[0]['images'][0]
print(f'First image: {img}')
print(f'Exists: {os.path.exists(img)}')
"
echo "=== Remap complete ==="
