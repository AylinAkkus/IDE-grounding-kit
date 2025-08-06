import os
import json
import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from tqdm import tqdm
from collections import defaultdict
import random
from huggingface_hub import HfApi, create_repo, upload_file
import tempfile
import dotenv

dotenv.load_dotenv()


def check_shard_exists(repo_id: str, dataset_name: str, shard_idx: int, api: HfApi) -> bool:
    """
    Check if a shard already exists in the HuggingFace repository.
    
    Args:
        repo_id: HuggingFace repository ID
        dataset_name: Name of the dataset
        shard_idx: Index of the shard to check
        api: HuggingFace API instance
        
    Returns:
        True if both JSON and zip files exist for this shard
    """
    try:
        # List files in the repository
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        
        # Construct expected file names
        json_filename = f"{dataset_name}_shard_{shard_idx:04d}.json"
        zip_filename = f"{dataset_name}_images_shard_{shard_idx:04d}.zip"
        
        # Check if both files exist in the expected paths
        json_path = f"{dataset_name}/{json_filename}"
        zip_path = f"{dataset_name}/{zip_filename}"
        
        return json_path in files and zip_path in files
    except Exception as e:
        # If we can't check (e.g., repo doesn't exist), assume shard doesn't exist
        print(f"Warning: Could not check if shard exists: {e}")
        return False


def build_image_sample_mappings(samples: List[Dict]) -> Tuple[Dict[str, List[int]], Dict[int, List[str]]]:
    """
    Build bidirectional mappings between images and samples.
    
    Args:
        samples: List of sample dictionaries
        
    Returns:
        Tuple of:
        - image_to_samples: Dict mapping image path to list of sample indices
        - sample_to_images: Dict mapping sample index to list of image paths
    """
    image_to_samples = defaultdict(list)
    sample_to_images = {}
    
    for idx, sample in enumerate(samples):
        images = sample.get("images", [])
        sample_to_images[idx] = images
        
        for image_path in images:
            image_to_samples[image_path].append(idx)
    
    return dict(image_to_samples), sample_to_images


def compute_shard_closure(
    initial_images: Set[str],
    image_to_samples: Dict[str, List[int]],
    sample_to_images: Dict[int, List[str]]
) -> Tuple[Set[str], Set[int]]:
    """
    Compute the closure of images and samples for a shard.
    Starting with a set of images, include all samples that use those images,
    then include all images used by those samples, and so on until we reach a fixed point.
    
    Args:
        initial_images: Initial set of image paths
        image_to_samples: Mapping from image to sample indices
        sample_to_images: Mapping from sample index to image paths
        
    Returns:
        Tuple of (final_images, final_samples)
    """
    images = set(initial_images)
    samples = set()
    
    # Keep expanding until we reach a fixed point
    while True:
        # Find all samples that use any of the current images
        new_samples = set()
        for image in images:
            new_samples.update(image_to_samples.get(image, []))
        
        # Find all images used by any of the samples
        new_images = set()
        for sample_idx in new_samples:
            new_images.update(sample_to_images.get(sample_idx, []))
        
        # Check if we've reached a fixed point
        if new_images == images and new_samples == samples:
            break
            
        images = new_images
        samples = new_samples
    
    return images, samples


def create_shard(
    samples: List[Dict],
    unique_images: List[str],
    image_base_dir: Path,
    shard_idx: int,
    temp_dir: Path,
    dataset_name: str
) -> Tuple[Path, Path]:
    """
    Create a shard with JSON and image zip files.
    
    Args:
        samples: List of samples for this shard
        unique_images: List of unique image paths in this shard
        image_base_dir: Base directory where images are stored
        shard_idx: Index of this shard
        temp_dir: Temporary directory for creating files
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (json_path, zip_path)
    """
    # Create JSON file
    json_filename = f"{dataset_name}_shard_{shard_idx:04d}.json"
    json_path = temp_dir / json_filename
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    # Create zip file with images
    zip_filename = f"{dataset_name}_images_shard_{shard_idx:04d}.zip"
    zip_path = temp_dir / zip_filename
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for image_path in tqdm(unique_images, desc=f"Zipping images for shard {shard_idx}"):
            # Construct full path to image
            # The image path might include a base directory name
            if os.path.sep in image_path:
                # Extract just the filename part after the base directory
                parts = image_path.split(os.path.sep)
                if len(parts) >= 2:
                    # Reconstruct path relative to image_base_dir
                    relative_path = os.path.join(*parts[1:])
                    full_image_path = image_base_dir / relative_path
                else:
                    full_image_path = image_base_dir / image_path
            else:
                full_image_path = image_base_dir / image_path
            
            if not full_image_path.exists():
                # Try without the base directory prefix
                base_name = os.path.basename(image_path)
                full_image_path = image_base_dir / base_name
            
            if full_image_path.exists():
                # Add to zip with the same relative path structure
                zf.write(full_image_path, arcname=image_path)
            else:
                print(f"Warning: Image not found: {full_image_path}")
    
    return json_path, zip_path


def upload_shard_to_hf(
    json_path: Path,
    zip_path: Path,
    repo_id: str,
    dataset_name: str,
    shard_idx: int,
    api: HfApi
):
    """
    Upload a shard (JSON and zip) to HuggingFace.
    
    Args:
        json_path: Path to JSON file
        zip_path: Path to zip file
        repo_id: HuggingFace repository ID
        dataset_name: Name of the dataset
        shard_idx: Index of this shard
        api: HuggingFace API instance
    """
    # Upload JSON file
    json_repo_path = f"{dataset_name}/{json_path.name}"
    upload_file(
        path_or_fileobj=str(json_path),
        path_in_repo=json_repo_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add {dataset_name} shard {shard_idx:04d} JSON"
    )
    
    # Upload zip file
    zip_repo_path = f"{dataset_name}/{zip_path.name}"
    upload_file(
        path_or_fileobj=str(zip_path),
        path_in_repo=zip_repo_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add {dataset_name} shard {shard_idx:04d} images"
    )


def process_dataset(
    json_file: Path,
    image_dir: Path,
    repo_id: str,
    dataset_name: str,
    images_per_shard: int = 10000,
    seed: Optional[int] = None,
    upload: bool = True,
    local_output_dir: Optional[Path] = None,
    force_reupload: bool = False
):
    """
    Process a dataset: shard it and optionally upload to HuggingFace.
    
    Args:
        json_file: Path to the JSON file with all samples
        image_dir: Directory containing images
        repo_id: HuggingFace repository ID
        dataset_name: Name of the dataset (for organizing in the repo)
        images_per_shard: Maximum images per shard
        seed: Random seed for shuffling
        upload: Whether to upload to HuggingFace
        local_output_dir: If provided, save shards locally instead of temp dir
        force_reupload: If True, reupload shards even if they already exist
    """
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"JSON file: {json_file}")
    print(f"Image directory: {image_dir}")
    print(f"Repository: {repo_id}")
    print(f"Images per shard: {images_per_shard}")
    if force_reupload:
        print("Force reupload: ENABLED - will reupload existing shards")
    
    # Load all samples
    with open(json_file, "r", encoding="utf-8") as f:
        all_samples = json.load(f)
    
    print(f"Total samples loaded: {len(all_samples)}")
    
    # Build mappings
    image_to_samples, sample_to_images = build_image_sample_mappings(all_samples)
    all_images = list(image_to_samples.keys())
    print(f"Total unique images: {len(all_images)}")
    
    # Calculate statistics
    samples_per_image = [len(samples) for samples in image_to_samples.values()]
    avg_samples_per_image = sum(samples_per_image) / len(samples_per_image) if samples_per_image else 0
    print(f"Average samples per image: {avg_samples_per_image:.2f}")
    
    images_per_sample = [len(images) for images in sample_to_images.values()]
    avg_images_per_sample = sum(images_per_sample) / len(images_per_sample) if images_per_sample else 0
    print(f"Average images per sample: {avg_images_per_sample:.2f}")
    
    # Shuffle the images
    if seed is not None:
        random.seed(seed)
    
    random.shuffle(all_images)
    
    # Initialize HuggingFace API (needed for checking existing shards)
    api = None
    if upload or not force_reupload:
        api = HfApi(token=os.getenv("HF_TOKEN"))
    
    # Create shards
    if upload:
        # Create repository if it doesn't exist
        try:
            create_repo(repo_id, repo_type="dataset", exist_ok=True)
            print(f"Repository {repo_id} ready")
        except Exception as e:
            print(f"Note: {e}")
    
    # Use temp directory or local output directory
    if local_output_dir:
        output_base = local_output_dir / dataset_name
        output_base.mkdir(parents=True, exist_ok=True)
        temp_dir = output_base
        use_temp = False
    else:
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = Path(temp_dir_obj.name)
        use_temp = True
    
    try:
        shard_idx = 0
        processed_images = set()
        processed_samples = set()
        shards_created = 0
        shards_skipped = 0
        
        for start_idx in range(0, len(all_images), images_per_shard):
            # Get the initial set of images for this shard
            end_idx = min(start_idx + images_per_shard, len(all_images))
            initial_images = set(all_images[start_idx:end_idx])
            
            # Skip images that have already been processed
            initial_images = initial_images - processed_images
            if not initial_images:
                continue
            
            # Compute the closure to get all related images and samples
            shard_images, shard_sample_indices = compute_shard_closure(
                initial_images,
                image_to_samples,
                sample_to_images
            )
            
            # Skip samples that have already been processed
            shard_sample_indices = shard_sample_indices - processed_samples
            if not shard_sample_indices:
                continue
            
            # Get the actual sample objects
            shard_samples = [all_samples[idx] for idx in sorted(shard_sample_indices)]
            shard_images_list = sorted(list(shard_images))
            
            print(f"\nCreating shard {shard_idx} with {len(shard_samples)} samples and {len(shard_images)} images")
            print(f"  Initial images requested: {len(initial_images)}")
            print(f"  Additional images needed for closure: {len(shard_images) - len(initial_images)}")
            
            # Check if shard already exists on HuggingFace
            skip_shard = False
            if upload and not force_reupload:
                if check_shard_exists(repo_id, dataset_name, shard_idx, api):
                    print(f"  Shard {shard_idx} already exists on HuggingFace, skipping...")
                    skip_shard = True
                    shards_skipped += 1
            
            if not skip_shard:
                # Create and upload the shard
                json_path, zip_path = create_shard(
                    shard_samples,
                    shard_images_list,
                    image_dir,
                    shard_idx,
                    temp_dir,
                    dataset_name
                )
                
                if upload:
                    print(f"Uploading shard {shard_idx} to HuggingFace...")
                    upload_shard_to_hf(
                        json_path,
                        zip_path,
                        repo_id,
                        dataset_name,
                        shard_idx,
                        api
                    )
                    
                    # Clean up if using temp directory
                    if use_temp:
                        os.remove(json_path)
                        os.remove(zip_path)
                else:
                    print(f"Saved shard {shard_idx} locally")
                
                shards_created += 1
            
            # Update processed sets
            processed_images.update(shard_images)
            processed_samples.update(shard_sample_indices)
            
            shard_idx += 1
        
        print(f"\nDataset processing complete!")
        print(f"Total shards created: {shards_created}")
        if shards_skipped > 0:
            print(f"Total shards skipped (already exist): {shards_skipped}")
        print(f"Total shards processed: {shard_idx}")
        print(f"Total images processed: {len(processed_images)}")
        print(f"Total samples processed: {len(processed_samples)}")
        
        # Check if we missed anything
        if len(processed_images) < len(all_images):
            print(f"Warning: {len(all_images) - len(processed_images)} images were not included in any shard")
        if len(processed_samples) < len(all_samples):
            print(f"Warning: {len(all_samples) - len(processed_samples)} samples were not included in any shard")
        
    finally:
        # Clean up temp directory if used
        if use_temp and 'temp_dir_obj' in locals():
            temp_dir_obj.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description='Shard and upload dataset to HuggingFace'
    )
    
    parser.add_argument('--json-file', '-j', required=True,
                        help='Path to the JSON file containing all samples')
    
    parser.add_argument('--image-dir', '-i', required=True,
                        help='Directory containing images')
    
    parser.add_argument('--repo-id', '-r', required=True,
                        help='HuggingFace repository ID (e.g., username/dataset-name)')
    
    parser.add_argument('--dataset-name', '-n', required=True,
                        help='Name of the dataset (used for organizing files in the repo)')
    
    parser.add_argument('--images-per-shard', '-s', type=int,
                        default=10000,
                        help='Maximum number of images per shard (default: 10000)')
    
    parser.add_argument('--seed', type=int,
                        default=None,
                        help='Random seed for shuffling (default: None)')
    
    parser.add_argument('--no-upload', action='store_true',
                        help='Do not upload to HuggingFace, just create shards locally')
    
    parser.add_argument('--local-output-dir', '-o',
                        help='Directory to save shards locally (default: temp directory)')
    
    parser.add_argument('--force-reupload', action='store_true',
                        help='Force reupload shards even if they already exist')
    
    args = parser.parse_args()
    
    # Validate paths
    json_file = Path(args.json_file)
    image_dir = Path(args.image_dir)
    
    if not json_file.exists():
        print(f"Error: JSON file '{json_file}' does not exist!")
        return
    
    if not image_dir.exists():
        print(f"Error: Image directory '{image_dir}' does not exist!")
        return
    
    # Process the dataset
    process_dataset(
        json_file=json_file,
        image_dir=image_dir,
        repo_id=args.repo_id,
        dataset_name=args.dataset_name,
        images_per_shard=args.images_per_shard,
        seed=args.seed,
        upload=not args.no_upload,
        local_output_dir=Path(args.local_output_dir) if args.local_output_dir else None,
        force_reupload=args.force_reupload
    )


if __name__ == "__main__":
    main()

"""
Example usage:
python shard_and_upload_dataset.py \
    --json-file dark_theme_cursor_icons_dataset/llamafactory_data.json \
    --image-dir dark_theme_cursor_icons_dataset/images \
    --repo-id mlfoundations-cua-dev/agent-perception_data \
    --dataset-name cursor_icons \
    --images-per-shard 10000 \
    --seed 42 \
    --no-upload

python shard_and_upload_dataset.py \
    --json-file C:/Users/aylin/Repos/IDE_use/IDE-grounding-kit/data/merged/merged_grounding_dataset.json \
    --image-dir C:/Users/aylin/Repos/IDE_use/IDE-grounding-kit/data/merged/images/
    --repo-id mlfoundations-cua-dev/agent-grounding-data \
    --dataset-name ide_icons_grounding \
    --images-per-shard 10000 \
    --seed 42 \
    --no-upload
"""