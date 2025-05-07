#!/usr/bin/env python3
import sys
import json
from pathlib import Path
from PIL import Image, ImageDraw

def main(directory):
    # Ensure the directory exists
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"❌ Error: {directory} is not a valid directory.")
        sys.exit(1)

    # Find the JSON, WebP, and HTML files
    json_files = list(dir_path.glob("*.json"))
    webp_files = list(dir_path.glob("*.webp"))
    html_files = list(dir_path.glob("*.html"))

    if len(json_files) != 1 or len(webp_files) != 1 or len(html_files) != 1:
        print("❌ Error: Directory must contain exactly one .json, one .webp, and one .html file.")
        sys.exit(1)

    json_path = json_files[0]
    webp_path = webp_files[0]
    prefix = json_path.stem  # Use the JSON file's name as the prefix

    # Load the analysis JSON
    data = json.load(open(json_path, encoding='utf-8'))
    caps = data['viewports'][0]['image_captioning']

    # Open the screenshot and prepare an RGBA overlay
    img = Image.open(webp_path).convert('RGBA')
    overlay = Image.new('RGBA', img.size)
    draw = ImageDraw.Draw(overlay)

    # Draw a red rectangle for each box with non-zero size
    for item in caps:
        x = item['bbox']['x']
        y = item['bbox']['y']
        w = item['bbox']['width']
        h = item['bbox']['height']
        if w > 0 and h > 0:
            draw.rectangle([x, y, x + w, y + h], outline='red', width=2)

    # Composite the overlay
    result = Image.alpha_composite(img, overlay)

    # Save the overlay image in the same directory
    out_path = dir_path / f"{prefix}-overlay.png"
    result.save(out_path)
    print(f"✅ Wrote overlay image to {out_path}")

    # Always show
    result.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python show_boxes.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    main(directory)
