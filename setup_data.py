"""
setup_data.py — Data folder scaffolding + guidance
====================================================
Run this FIRST before train_all.py to create the required folder structure
and get instructions on where to download monument images.

Usage:
    python setup_data.py
"""

import os
from pathlib import Path

STYLE_CLASSES = [
    "Dravidian",
    "Nagara",
    "Vesara",
    "Islamic",
    "Buddhist",
    "Colonial",
    "Rock_Cut",
    "Stepwell",
]

def create_structure():
    base = Path("data/raw")

    # Style classifier folders (one per class)
    for style in STYLE_CLASSES:
        folder = base / style
        folder.mkdir(parents=True, exist_ok=True)
        # Place a .gitkeep so git tracks the empty folder
        (folder / ".gitkeep").touch()

    # Inpainting / GAN folder
    (base / "all_monuments").mkdir(parents=True, exist_ok=True)
    (base / "all_monuments" / ".gitkeep").touch()

    print("✅ Folder structure created:\n")
    print("  data/raw/")
    for style in STYLE_CLASSES:
        count = len(list((base / style).glob("*.jpg"))) + \
                len(list((base / style).glob("*.png")))
        status = f"  ← {count} images found" if count else "  ← EMPTY — needs images"
        print(f"    {style}/{status}")

    inpaint_count = len(list((base / "all_monuments").glob("*.jpg"))) + \
                    len(list((base / "all_monuments").glob("*.png")))
    inpaint_status = f"  ← {inpaint_count} images found" if inpaint_count else "  ← EMPTY — needs images"
    print(f"    all_monuments/{inpaint_status}")


def print_download_guide():
    print("\n" + "="*60)
    print("  HOW TO GET MONUMENT IMAGES")
    print("="*60)

    print("""
─── Option 1: Wikimedia Commons (Free, recommended) ──────────
Search for each architectural style and download images.
URLs to start from:

  Dravidian : https://commons.wikimedia.org/wiki/Category:Dravidian_architecture
  Nagara    : https://commons.wikimedia.org/wiki/Category:Nagara_architecture
  Islamic   : https://commons.wikimedia.org/wiki/Category:Indo-Islamic_architecture
  Buddhist  : https://commons.wikimedia.org/wiki/Category:Buddhist_architecture_in_India
  Colonial  : https://commons.wikimedia.org/wiki/Category:Colonial_architecture_in_India
  Rock_Cut  : https://commons.wikimedia.org/wiki/Category:Rock-cut_architecture_in_India

  Tip: Use the 'Download all' option or a tool like 'gallery-dl':
       pip install gallery-dl
       gallery-dl "https://commons.wikimedia.org/wiki/Category:Dravidian_architecture"

─── Option 2: Google Images (quick scraping) ─────────────────
  pip install icrawler
  Then run:  python download_images.py   (created below)

─── Option 3: Kaggle Datasets ────────────────────────────────
  Search for "Indian monuments", "temple architecture India"
  on https://www.kaggle.com/datasets

─── Minimum required ─────────────────────────────────────────
  Style Classifier : ~50 images per class  (400 total minimum)
  Segmenter / GAN  : 200+ images in all_monuments/
                     (can copy all raw images there too)

─── Place images here ────────────────────────────────────────
  data/raw/Dravidian/    ← .jpg or .png files
  data/raw/Nagara/
  ... etc
  data/raw/all_monuments/  ← any clean monument images
""")


def create_icrawler_script():
    """Create a helper script to auto-download images using icrawler."""
    script = '''"""
download_images.py — Auto-download monument images using icrawler
=================================================================
Install first:  pip install icrawler

This downloads ~60 images per style class from Bing Image Search.
You may need to verify/clean images manually afterwards.
"""

from icrawler.builtin import BingImageCrawler
from pathlib import Path

QUERIES = {
    "Dravidian" : "Dravidian temple architecture India",
    "Nagara"    : "Nagara shikhara temple North India",
    "Vesara"    : "Vesara style temple Karnataka",
    "Islamic"   : "Indo-Islamic mosque architecture India",
    "Buddhist"  : "Buddhist stupa monastery India",
    "Colonial"  : "British colonial architecture India",
    "Rock_Cut"  : "rock cut cave temple India Ellora Ajanta",
    "Stepwell"  : "Indian stepwell vav architecture",
}

MAX_NUM = 60   # images per class — increase if you want more

for style, query in QUERIES.items():
    save_dir = f"data/raw/{style}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"\\nDownloading {MAX_NUM} images for: {style}")
    crawler = BingImageCrawler(storage={"root_dir": save_dir})
    crawler.crawl(keyword=query, max_num=MAX_NUM)

print("\\n✅ Download complete! Check data/raw/ for images.")
print("   Run: python setup_data.py   to see counts per class.")
'''
    with open("download_images.py", "w") as f:
        f.write(script)
    print("\n✅ Created download_images.py")
    print("   To auto-download images:  pip install icrawler && python download_images.py")


def check_ready():
    base = Path("data/raw")
    all_ok = True
    print("\n" + "="*60)
    print("  READINESS CHECK")
    print("="*60)
    for style in STYLE_CLASSES:
        folder = base / style
        count = len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.png")))
        status = "✅" if count >= 10 else ("⚠️  low" if count > 0 else "❌ empty")
        print(f"  {style:<15} {count:>4} images  {status}")
        if count == 0:
            all_ok = False

    inpaint = base / "all_monuments"
    count = len(list(inpaint.glob("*.jpg"))) + len(list(inpaint.glob("*.png")))
    status = "✅" if count >= 50 else ("⚠️  low" if count > 0 else "❌ empty")
    print(f"  {'all_monuments':<15} {count:>4} images  {status}")

    print()
    if all_ok:
        print("  ✅ Ready to train! Run:  python train_all.py --stage all")
    else:
        print("  ❌ Add images to the empty folders first, then run train_all.py")
    return all_ok


if __name__ == "__main__":
    create_structure()
    print_download_guide()
    create_icrawler_script()
    check_ready()
