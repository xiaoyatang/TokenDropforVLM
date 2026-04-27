import os
import json

JSON_PATH = "./playground/data/llava_v1_5_mix665k.json"
IMAGE_ROOT = "./playground/data"

total = 0
gqa_count = 0
missing_count = 0
sample_limit = 10
checked_limit = 5000

samples = []

def stream_json_array(file_path):
    """逐个yield JSON数组里的元素（避免OOM）"""
    with open(file_path, 'r') as f:
        buf = ""
        depth = 0
        in_string = False

        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            buf += chunk

            i = 0
            start = None
            while i < len(buf):
                c = buf[i]

                if c == '"' and (i == 0 or buf[i-1] != '\\'):
                    in_string = not in_string

                if not in_string:
                    if c == '{':
                        if depth == 0:
                            start = i
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0 and start is not None:
                            obj = buf[start:i+1]
                            yield json.loads(obj)
                            start = None

                i += 1

            # 保留未处理部分
            if depth == 0:
                buf = ""
            else:
                buf = buf[start:] if start is not None else buf


# ===== 主逻辑 =====
for item in stream_json_array(JSON_PATH):
    total += 1

    image_path = item.get("image", "")

    if "gqa" in image_path:
        gqa_count += 1

        if len(samples) < sample_limit:
            samples.append(image_path)

        if gqa_count <= checked_limit:
            full_path = os.path.join(IMAGE_ROOT, image_path)
            if not os.path.exists(full_path):
                print(f"[MISSING] {full_path}")
                missing_count += 1


print("\n===== SUMMARY =====")
print(f"Total samples: {total}")
print(f"GQA samples: {gqa_count}")
print(f"Missing (checked first {checked_limit}): {missing_count}")

print("\n===== SAMPLE PATHS =====")
for s in samples:
    print(s)