import json
from pathlib import Path

def get_grounding_task_message(line):
    x1, y1, x2, y2 = line["bbox_normalized"]
    bbox_center_x = int(round((x1 + x2) / 2))
    bbox_center_y = int(round((y1 + y2) / 2))
    assistant_answer = f"Action: (start_box='<|box_start|>({bbox_center_x},{bbox_center_y})<|box_end|>')"
    PROMPT = """<image>{instruction}"""
    msg = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT.format(instruction=line["function"])
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": assistant_answer
                }
            ]
        }
    ]
    return msg

def postprocess_to_grounding_dataset(input_file, theme):
    """
    Postprocess the grounding dataset and save to output folder.
    """
    base_path = Path(input_file).parent
    output_file = base_path / (f"{theme}_grounding_dataset.json")
    original_data = []
    with open(input_file, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                original_data.append(json.loads(line))

    print(f"Original dataset size: {len(original_data)}")

    # Debug: Print first few records to see structure
    if original_data:
        print("Sample record structure:")
        sample_record = original_data[0]
        print(f"Keys: {list(sample_record.keys())}")
        print(f"Function value: {sample_record.get('function')}")
        print(f"Function type: {type(sample_record.get('function'))}")

    # Create doubled dataset
    samples = []
    
    valid_functions_count = 0
    null_functions_count = 0

    for record in original_data:
        # Skip records with null function
        if record.get('function') is None:
            null_functions_count += 1
            continue
        else:
            valid_functions_count += 1

        
        # Grounding task
        sample = {
            "task_type": "grounding",
            "id": theme + "_" + str(record["backend_node_id"]),
            "messages": get_grounding_task_message(record),
            "images": [(Path("images") / f"{theme}.png").as_posix()],
            "bbox": [int(round(x)) for x in record["bbox_normalized"]],
            "original_bbox": [int(round(x)) for x in record["bbox_unnormalized"]],
            "instruction": record["function"]
        }
        
        samples.append(sample)

    print(f"Valid functions found: {valid_functions_count}")
    print(f"Null functions skipped: {null_functions_count}")
    print(f"Final dataset size: {len(samples)}")

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"Saved to {output_file}")