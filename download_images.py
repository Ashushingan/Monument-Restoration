"""
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
    print(f"\nDownloading {MAX_NUM} images for: {style}")
    crawler = BingImageCrawler(storage={"root_dir": save_dir})
    crawler.crawl(keyword=query, max_num=MAX_NUM)

print("\n✅ Download complete! Check data/raw/ for images.")
print("   Run: python setup_data.py   to see counts per class.")
