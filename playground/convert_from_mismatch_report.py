import json
from pathlib import Path
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPORT_PATH = Path("/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/playground/missing_images_ext_mismatch_report.json")
DATA_ROOT = Path("/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/playground/data")

with open(REPORT_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

converted = []
skipped = []
failed = []

def convert_to_jpg(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        rgb = im.convert("RGB")
        rgb.save(dst, format="JPEG", quality=95)

for r in records:
    image_in_json = r["image_in_json"]                      # e.g. ocr_vqa/images/898796008.jpg
    candidates = r.get("same_stem_different_ext", [])

    if not candidates:
        failed.append({
            "image_in_json": image_in_json,
            "reason": "no alternative extension found in report"
        })
        continue

    src = DATA_ROOT / candidates[0]                        # e.g. .../ocr_vqa/images/898796008.gif
    dst = DATA_ROOT / image_in_json                        # e.g. .../ocr_vqa/images/898796008.jpg

    if dst.exists():
        skipped.append(str(dst))
        continue

    if not src.exists():
        failed.append({
            "image_in_json": image_in_json,
            "src": str(src),
            "reason": "source file missing"
        })
        continue

    try:
        convert_to_jpg(src, dst)
        print(f"[OK] {src.name} -> {dst.name}")
        converted.append({
            "src": str(src),
            "dst": str(dst)
        })
    except Exception as e:
        print(f"[FAIL] {src} -> {dst}: {type(e).__name__}: {e}")
        failed.append({
            "image_in_json": image_in_json,
            "src": str(src),
            "dst": str(dst),
            "error_type": type(e).__name__,
            "error": str(e)
        })

out_dir = REPORT_PATH.parent
with open(out_dir / "converted_from_mismatch_report.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

with open(out_dir / "failed_from_mismatch_report.json", "w", encoding="utf-8") as f:
    json.dump(failed, f, ensure_ascii=False, indent=2)

print("\nDone")
print("converted:", len(converted))
print("skipped existing:", len(skipped))
print("failed:", len(failed))