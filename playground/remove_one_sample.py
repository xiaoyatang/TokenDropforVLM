import json

src = "/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/playground/data/llava_v1_5_mix665k_backup.json"
dst = "/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/playground/data/llava_v1_5_mix665k.json"
target = "ocr_vqa/images/689852649.jpg"

with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)

remove_idx = None
for i, x in enumerate(data):
    if x.get("image") == target:
        remove_idx = i
        break

if remove_idx is None:
    print("target not found")
else:
    del data[remove_idx]
    print("removed 1 sample at index", remove_idx)

with open(dst, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")

print("saved to", dst)