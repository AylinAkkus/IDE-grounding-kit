import argparse
import asyncio
import os
import shutil
from pathlib import Path

# Import the pipeline functions
from grounding import extract_icons, annotate_icons_with_instructions
from postprocessing import postprocess_to_grounding_dataset


def main(args):
    if args.mode == "perception":
        raise NotImplementedError("Perception mode is not implemented yet")
    
    elif args.mode == "grounding":
        print(f"Starting grounding dataset generation for theme: {args.theme_name}")
        
        # Create output directory structure
        output_dir = Path(args.output_dir) / args.theme_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        icons_data_path = output_dir / "icons_data.json"
        bbox_images_dir = output_dir / "bbox_images"
        images_dir = output_dir / "images"
        annotated_data_path = output_dir / "annotated_icons.jsonl"
        
        try:
            # Skip icon extraction and if the file already exist
            if not args.force_rerun and os.path.exists(icons_data_path):
                print(f"Skipping icon extraction and annotation as {icons_data_path} already exists")
                print(f"To force rerun the pipeline, use --force-rerun")
            
            else:
                # Step 1: Extract icons
                print("\nExtracting icons from browser...")
                # Create a simple args object for extract_icons function
                class ExtractionArgs:
                    def __init__(self, cdp_url, output_dir, theme_name):
                        self.cdp_url = cdp_url
                        self.output_dir = str(output_dir)
                        self.theme_name = theme_name
                
                extraction_args = ExtractionArgs(args.web_socket_url, output_dir, args.theme_name)
                asyncio.run(extract_icons(extraction_args))
                # Verify icons were extracted
                if not icons_data_path.exists():
                    raise FileNotFoundError(f"Icon extraction failed - {icons_data_path} not found")
            
            if not args.force_rerun and os.path.exists(annotated_data_path):
                print(f"Skipping icon annotation as {annotated_data_path} already exists")
                print(f"To force rerun the pipeline, use --force-rerun")
            else:
                # Step 2: Annotate icons with instructions
                print("\nAnnotating icons with instructions...")
                annotate_icons_with_instructions(
                    data_path=str(icons_data_path),
                    images_dir=str(bbox_images_dir), 
                    output_path=str(annotated_data_path)
                )
                
                # Verify annotation was completed
                if not annotated_data_path.exists():
                    raise FileNotFoundError(f"Icon annotation failed - {annotated_data_path} not found")
            
            # Step 3: Postprocess to grounding dataset
            print("\nCreating grounding dataset...")
            postprocess_to_grounding_dataset(
                input_file=str(annotated_data_path),
                theme=args.theme_name
            )
            
            print(f"\nDataset generation completed successfully!")
            print(f"Output directory: {output_dir}")
            print(f"Final dataset: {args.theme_name}_grounding_dataset.json")
            
        except Exception as e:
            print(f"\nError during dataset generation: {str(e)}")
            raise

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--theme-name", type=str, required=True, help="The name of theme of the IDE")
    argparser.add_argument("--web-socket-url", type=str, required=True, help="The URL of the web socket")
    argparser.add_argument("--output-dir", type=str, default="data", help="The output directory")
    argparser.add_argument("--force-rerun", action="store_true", help="Force rerun the pipeline")
    argparser.add_argument("--mode", choices=["grounding", "perception"], required=True, help="The type of data to generate")
    args = argparser.parse_args()

    main(args)

"""
python main.py \
    --mode grounding \
    --theme-name dark_theme \
    --web-socket-url ws://127.0.0.1:9222/devtools/browser/5300fb59-8b33-4d45-ac9a-137fa1ee4a55
"""