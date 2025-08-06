"""
Icon Function Annotation Script

This script uses Anthropic's Claude API to annotate the function of GUI icons as an instruction.
GUI icons are collected from the browser DOM tree.
"""
from anthropic import Anthropic
import os
import json
from PIL import Image
import base64
from pathlib import Path
import re
from tqdm import tqdm
import dotenv

# Initialize Anthropic client with API key from environment
# For security, it's better to use: export ANTHROPIC_API_KEY="your-key-here"
dotenv.load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

PROMPT = """You are a helpful assistant trained to identify and describe the function of GUI icons in the Cursor IDE.

For each task, you are given:
- An image showing a red bounding box around a UI element
- The element’s ARIA label (if available)
- Its full class name

Your job:
1. Decide whether the highlighted element is a **GUI icon**.
2. If it **is** a GUI icon, write an imperative instruction that would prompt a user to **click** that icon, using this format:
   <instruction>...</instruction>
3. If it is **not** a GUI icon, output:
   <not_a_gui_icon>

The instruction should be clear, concise, and action-oriented—describing exactly what would happen when the icon is clicked.

**Example:**
Aria label: "Toggle Primary Side Bar (Ctrl+B)"  
Class name: "action-label checked codicon codicon-panel-left"  
Assistant: <instruction>Toggle the primary side bar.</instruction>

---

Now annotate the following:

Aria label: {aria_label}  
Class name: {class_name}
"""

def encode_image_to_base64(image_path):
    """Convert image to base64 format for Anthropic API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_function_from_response(response_text):
    """Extract function from Claude's response using regex"""
    # Look for <function>...</function> tags
    function_match = re.search(r'<instruction>(.*?)</instruction>', response_text, re.DOTALL)
    if function_match:
        return function_match.group(1).strip()
    
    # Look for <not_a_gui_icon> tag
    if '<not_a_gui_icon>' in response_text:
        return None
    
    # If no tags found, return the response as-is (fallback)
    return response_text.strip()

def annotate_icons_with_instructions(data_path, images_dir, output_path):
    """Annotate icons using Anthropic API and save results"""
    
    # Load the icons data
    data_path = Path(data_path)
    images_dir = Path(images_dir)
    output_path = Path(output_path)
    
    print(f"Loading icons data from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        icons_data = json.load(f)
    
    print(f"Found {len(icons_data)} icons to annotate")
    
    # Clear output file before starting
    if output_path.exists():
        output_path.unlink()
    
    # Create annotated data list
    annotated_data = []
    
    # Process each icon
    for item in tqdm(icons_data, desc="Annotating icons"):
        # Copy all original data
        annotated_item = item.copy()
        
        try:
            # Get image path
            image_filename = item["images"][0]
            image_path = images_dir / image_filename
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                annotated_item["function"] = None
                annotated_data.append(annotated_item)
                
                # Save to JSONL even for missing images
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(annotated_item, ensure_ascii=False) + "\n")
                continue
            
            # Encode image to base64
            base64_image = encode_image_to_base64(image_path)
            
            # Make API call to Anthropic
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=200,
                messages=[{
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT.format(
                                aria_label=item["aria_label"], 
                                class_name=item["classes"]
                            )
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image
                            }
                        }
                    ]
                }]
            )
            
            # Extract function from response
            response_text = response.content[0].text
            print(response_text)
            function = extract_function_from_response(response_text)
            annotated_item["function"] = function
            
        except Exception as e:
            print(f"Error processing item {item.get('backend_node_id', 'unknown')}: {str(e)}")
            annotated_item["function"] = None
        
        # Add to data list
        annotated_data.append(annotated_item)
        
        # Save as JSONL (append each item as a new line)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(annotated_item, ensure_ascii=False) + "\n")
    
    # Print summary
    functions_found = sum(1 for item in annotated_data if item["function"] is not None)
    print(f"\nAnnotation complete!")
    print(f"Total icons: {len(annotated_data)}")
    print(f"GUI icons with functions: {functions_found}")
    print(f"Non-GUI icons or errors: {len(annotated_data) - functions_found}")
    print(f"Results saved as JSONL format to: {output_path}")
    print("Each line in the JSONL file is a complete JSON object with all original data + function annotation")
