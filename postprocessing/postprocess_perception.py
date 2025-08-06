import json

def get_bbox_to_func_message(line):
    bbox_string = "<bbox> {x1}, {y1}, {x2}, {y2} </bbox>".format(x1=line["bbox_normalized"][0], y1=line["bbox_normalized"][1], x2=line["bbox_normalized"][2], y2=line["bbox_normalized"][3])
    PROMPT = """<image>What function does the GUI element with the following bounding box have?:
    {bbox_string}"""
    msg = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT.format(bbox_string=bbox_string)
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": line["function"]
                }
            ]
        }
    ]
    return msg

def get_func_to_bbox_message(line):
    PROMPT = """<image>Find the bounding box of the GUI element with the following function: {function}"""
    bbox_string = "<bbox> {x1}, {y1}, {x2}, {y2} </bbox>".format(x1=line["bbox_normalized"][0], y1=line["bbox_normalized"][1], x2=line["bbox_normalized"][2], y2=line["bbox_normalized"][3])
    msg = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT.format(function=line["function"])
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": bbox_string
                }
            ]
        }
    ]
    return msg

def postprocess_to_perception_dataset(input_file, theme):
    """
    Postprocess the perception dataset to create a doubled dataset.
    """
    output_file = theme + "_perception_dataset.json"
    original_data = []
    with open(input_file, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                original_data.append(json.loads(line))

    print(f"Original dataset size: {len(original_data)}")

    # Create doubled dataset
    samples = []

    for record in original_data:
        # Skip records with null function
        if record.get('function') is None:
            continue
        
        # Task 1: function_to_bbox - given function, predict bbox
        function_to_bbox = {
            "task_type": "function_to_bbox",
            "id": record["backend_node_id"],
            "messages": get_func_to_bbox_message(record),
            "images": "images/dark_theme.png"
        }
        
        # Task 2: bbox_to_function - given bbox, predict function  
        bbox_to_function = {
            "task_type": "bbox_to_function",
            "id": theme + "_" + str(record["backend_node_id"]),
            "messages": get_bbox_to_func_message(record),
            "images": "images/dark_theme.png"
        }
        
        samples.extend([function_to_bbox, bbox_to_function])

        print(f"Dataset size: {len(samples)}")

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"Saved to {output_file}")