import argparse
import yaml
import logging
import sys
import os
import json
from glob import glob

# Add src to the Python path (handle both local and Kaggle)
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.pipeline import SearchPipeline

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] [%(name)s] - %(message)s',
                    handlers=[
                        logging.FileHandler("pipeline.log"),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Drone Search and Rescue Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    # --- Load Config ---
    try:
        config_path = os.path.abspath(args.config)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return

    # Resolve relative paths in config to be relative to the config file location
    config_base = os.path.dirname(config_path)
    def resolve_path(p: str) -> str:
        if not isinstance(p, str):
            return p
        # if it's already absolute, return as-is
        if os.path.isabs(p):
            return p
        # otherwise join with config base
        return os.path.normpath(os.path.join(config_base, p))

    # Common top-level keys that may contain paths
    if 'video_input_dir' in config:
        config['video_input_dir'] = resolve_path(config['video_input_dir'])
    if 'submission_output_path' in config:
        config['submission_output_path'] = resolve_path(config['submission_output_path'])
    if 'reference_images' in config:
        config['reference_images'] = [resolve_path(p) for p in config['reference_images']]
    if 'finder' in config and isinstance(config['finder'], dict):
        if 'checkpoint' in config['finder']:
            config['finder']['checkpoint'] = resolve_path(config['finder']['checkpoint'])

    # --- Find all videos to process ---
    video_dir = config['video_input_dir']
    output_path = config['submission_output_path']
    
    # Find all common video formats
    video_paths = glob(os.path.join(video_dir, "*.mp4")) + \
                  glob(os.path.join(video_dir, "*.avi")) + \
                  glob(os.path.join(video_dir, "*.mov"))
    
    if not video_paths:
        logger.error(f"No videos found in {video_dir}")
        return
        
    logger.info(f"Found {len(video_paths)} videos to process.")

    # --- This list will hold the final JSON content ---
    all_submissions = []

    for video_path in video_paths:
        video_id = os.path.basename(video_path)
        logger.info(f"--- Starting processing for {video_id} ---")
        
        # Create a *new* config for this specific video
        video_config = config.copy()
        video_config['video_input_path'] = video_path
        
        try:
            # Create a *fresh* pipeline for each video
            # This resets the tracker and all object IDs
            pipeline = SearchPipeline(video_config)
            
            # Run the pipeline and get the submission dict for this video
            video_result = pipeline.run_on_video()
            
            all_submissions.append(video_result)
            
        except Exception as e:
            logger.critical(f"A critical error occurred on {video_id}: {e}", exc_info=True)
            # Add an empty result for this video per requirement
            all_submissions.append({
                "video_id": video_id,
                "detections": []
            })

    # --- Save the final submission file ---
    try:
        with open(output_path, 'w') as f:
            json.dump(all_submissions, f, indent=4)
        logger.info(f"Submission file saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save submission file: {e}")

if __name__ == "__main__":
    main()