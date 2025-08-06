"""
Icon Extractor Script

This script connects to a Chrome browser via Chrome DevTools Protocol (CDP),
extracts all icons from the DOM tree, saves them as individual images, and
creates a JSON file with metadata about each icon.

Usage:
    python connect_cdp.py --cdp_url "ws://127.0.0.1:9222/devtools/browser/..." --output_dir "./output"

Requirements:
    - Chrome browser running with --remote-debugging-port=9222
    - playwright, Pillow, and dom_optimized modules installed
"""

import asyncio
import logging
import json
from pathlib import Path
from playwright.async_api import async_playwright
import argparse

# Import the DOM service and related classes
from dom import DOMService, DOMElementNode, DOMTree
from PIL import Image, ImageDraw
from tqdm import tqdm


def take_cropped_screenshot(el: DOMElementNode, full: Image.Image) -> None:
    if not el.bounding_box:
        return
    x, y, w, h = map(int, (el.bounding_box["x"], el.bounding_box["y"],
                            el.bounding_box["width"], el.bounding_box["height"]))
    full.crop((x, y, x + w, y + h)).save(f"test/{el.node_id}.png")

def is_icon(el: DOMElementNode) -> bool:
    """
    Identify if an element is an icon based on class names.
    This function looks for common icon class patterns.
    """
    class_attr = el.attributes.get("class", "")
    # Look for common icon class patterns
    icon_indicators = ["codicon", "icon", "fa-", "material-icons", "glyphicon"]
    return any(indicator in class_attr for indicator in icon_indicators) and el.is_visible

def get_viewport_size(image: Image.Image) -> tuple:
    return image.size

def take_bbox_screenshot(path: Path, el: DOMElementNode, full: Image.Image) -> None:
    if not el.bounding_box:
        return
    x, y, w, h = map(int, (el.bounding_box["x"], el.bounding_box["y"],
                           el.bounding_box["width"], el.bounding_box["height"]))
    img = full.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
    backend_node_id = str(el.backend_node_id)
    img.save(path / f"{backend_node_id}.png")

def bbox_to_list(bbox: dict) -> list:
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    # We convert to x1, y1, x2, y2 format	
    return [x, y, x + w, y + h]

def normalize_bbox(bbox: list, viewport_size: tuple) -> list:
    x1, y1, x2, y2 = bbox
    return [(x1 / viewport_size[0])*1000, (y1 / viewport_size[1])*1000, (x2 / viewport_size[0])*1000, (y2 / viewport_size[1])*1000]

def save_icons_and_create_json(icons, bbox_images_dir, full_image, viewport_size):
    """
    Save icons to images folder and create JSON with backend_node_id, aria-label, and classes
    """
    
    # Prepare JSON data
    icons_data = []
    
    for icon in tqdm(icons, desc="Processing icons", total=len(icons)):
        # Save icon image with backend_node_id as filename
        if icon.bounding_box:
            take_bbox_screenshot(bbox_images_dir, icon, full_image)
            bbox_unnormalized = bbox_to_list(icon.bounding_box)
            bbox_normalized = normalize_bbox(bbox_unnormalized, viewport_size)
            
            # Collect data for JSON
            icon_data = {
                "backend_node_id": icon.backend_node_id,
                "bbox_unnormalized": bbox_unnormalized,
                "bbox_normalized": bbox_normalized,
                "aria_label": icon.attributes.get("aria-label", ""),
                "classes": icon.attributes.get("class", ""),
                "images": [str(icon.backend_node_id) + ".png"]
            }
            icons_data.append(icon_data)
    
    # Save JSON data
    json_path = Path(bbox_images_dir.parent) / "icons_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(icons_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(icons_data)} icons to {bbox_images_dir}")
    print(f"Saved JSON data to {json_path}")



async def extract_icons(args):
    """Main function to extract icons from browser tab"""
    # Set up logging to see DOM service debug info
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create images directories
    bbox_images_dir = Path(args.output_dir) / "bbox_images"
    bbox_images_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        async with async_playwright() as p:
            print("Connecting to CDP...")
            browser = await p.chromium.connect_over_cdp(args.cdp_url)
            
            if not browser.contexts:
                raise Exception("No browser contexts found. Make sure Chrome is running with a tab open.")
            
            context = browser.contexts[0]
            
            if not context.pages:
                raise Exception("No pages found in the browser context.")
                
            page = context.pages[0]  # Attach to the first open page
            cdp_session = await context.new_cdp_session(page)

            # Create DOM service instance (requires context and cdp_session)
            dom_service = DOMService(page, context, cdp_session)
            
            # Build DOM tree (enrichment happens automatically)
            print("Building and enriching DOM tree...")
            dom_tree: DOMTree = await dom_service.build_dom_tree()

            # Find all icons
            icons = dom_tree.root.find_all(is_icon)
            print(f"Found {len(icons)} icons")

            if not icons:
                print("No icons found. The page might not contain recognizable icon elements.")
                return

            # Take a screenshot of the full page
            print("Taking full page screenshot...")
            await page.screenshot(path=images_dir / f"{args.theme_name}.png", full_page=True)
            full = Image.open(images_dir / f"{args.theme_name}.png")
            
            # Get viewport size
            viewport_size = get_viewport_size(full)

            # Save icons and create JSON
            save_icons_and_create_json(icons, bbox_images_dir, full, viewport_size)
            
            print("Icon extraction completed successfully!")

            await browser.close()
            
    except Exception as e:
        logger.error(f"Error during icon extraction: {str(e)}")
        print(f"Error: {str(e)}")
        raise

