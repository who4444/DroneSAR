"""
Kaggle Runner Script for Drone SAR Pipeline
============================================

Usage on Kaggle:
    python kaggle_run.py --dataset <dataset-folder-name-in-/kaggle/input>

If --dataset is omitted, the script auto-detects if exactly one folder exists under /kaggle/input.
Output submission file will be written to /kaggle/working/submission.json

Features:
- Resolves all hardcoded paths to work with Kaggle's directory structure
- Handles both local and Kaggle execution environments
- Auto-downloads YOLO model if not present
- Provides detailed error messages for debugging
"""
import os
import sys
import yaml
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

KAGGLE_INPUT_ROOT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"
LOCAL_MODE = not os.path.exists(KAGGLE_INPUT_ROOT)


def find_kaggle_base(dataset_name: str = None) -> str:
    """Find the base dataset directory on Kaggle."""
    if LOCAL_MODE:
        logger.warning("Not running on Kaggle. Using local paths.")
        return None
    
    if dataset_name:
        candidate = os.path.join(KAGGLE_INPUT_ROOT, dataset_name)
        if os.path.isdir(candidate):
            logger.info(f"Using dataset: {dataset_name}")
            return candidate
        logger.error(f"Dataset folder not found: {candidate}")
        return None
    
    # Auto-detect: if only one folder under /kaggle/input, use it
    if not os.path.isdir(KAGGLE_INPUT_ROOT):
        return None
    
    entries = [d for d in os.listdir(KAGGLE_INPUT_ROOT) 
               if os.path.isdir(os.path.join(KAGGLE_INPUT_ROOT, d)) and not d.startswith('.')]
    
    if len(entries) == 1:
        base = os.path.join(KAGGLE_INPUT_ROOT, entries[0])
        logger.info(f"Auto-detected dataset: {entries[0]}")
        return base
    
    if len(entries) > 1:
        logger.error(f"Multiple datasets found: {entries}. Please specify with --dataset")
        return None
    
    logger.error("No datasets found under /kaggle/input")
    return None


def resolve_and_write_config(base_config_path: str, kaggle_base: str = None) -> str:
    """
    Load config, resolve all paths to Kaggle structure, and write output config.
    
    Path resolution strategy:
    1. If absolute path: use as-is
    2. If file exists in kaggle_base/<relative_path>: use that
    3. Otherwise: resolve relative to config file location
    """
    config_path = os.path.abspath(base_config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config_base = os.path.dirname(config_path)
    
    def resolve_path(p):
        """Resolve a single path."""
        if not isinstance(p, str):
            return p
        
        # Already absolute
        if os.path.isabs(p):
            if os.path.exists(p):
                return p
            logger.warning(f"Absolute path does not exist: {p}")
            return p
        
        # Try kaggle_base first
        if kaggle_base:
            rel = p.lstrip("./")
            cand = os.path.join(kaggle_base, rel)
            if os.path.exists(cand):
                logger.info(f"Resolved {p} -> {cand} (Kaggle)")
                return cand
        
        # Resolve relative to config file
        resolved = os.path.normpath(os.path.join(config_base, p))
        if os.path.exists(resolved):
            logger.info(f"Resolved {p} -> {resolved} (local)")
            return resolved
        
        # Return as-is if it doesn't exist yet (might be created later)
        logger.warning(f"Path does not exist (will create later): {p}")
        return resolved
    
    # Resolve all path fields in config
    logger.info("Resolving configuration paths...")
    
    if 'video_input_dir' in config:
        config['video_input_dir'] = resolve_path(config['video_input_dir'])
    
    if 'submission_output_path' in config:
        if LOCAL_MODE:
            config['submission_output_path'] = resolve_path(config['submission_output_path'])
        else:
            # On Kaggle, always write to /kaggle/working
            config['submission_output_path'] = os.path.join(KAGGLE_WORKING, 'submission.json')
            logger.info(f"Submission will be written to: {config['submission_output_path']}")
    
    if 'reference_images' in config and isinstance(config['reference_images'], list):
        config['reference_images'] = [resolve_path(p) for p in config['reference_images']]
    
    if 'finder' in config and isinstance(config['finder'], dict):
        if 'checkpoint' in config['finder']:
            config['finder']['checkpoint'] = resolve_path(config['finder']['checkpoint'])
    
    # Write the resolved config to working directory
    if LOCAL_MODE:
        out_cfg_path = os.path.join(os.getcwd(), 'config_for_run.yaml')
    else:
        os.makedirs(KAGGLE_WORKING, exist_ok=True)
        out_cfg_path = os.path.join(KAGGLE_WORKING, 'config_for_kaggle.yaml')
    
    with open(out_cfg_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    logger.info(f"Resolved config written to: {out_cfg_path}")
    return out_cfg_path


def main():
    parser = argparse.ArgumentParser(
        description="Run Drone SAR pipeline on Kaggle or locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Local run:        python kaggle_run.py
  Kaggle run:       python kaggle_run.py --dataset my-dataset
  Custom config:    python kaggle_run.py --config /path/to/config.yaml --dataset my-dataset
        """
    )
    parser.add_argument('--dataset', type=str, default=None, 
                        help='Name of dataset folder under /kaggle/input')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to base config file (default: configs/config.yaml)')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Drone SAR Pipeline - Kaggle Runner")
    logger.info("=" * 60)
    
    # Find Kaggle base if running on Kaggle
    kaggle_base = find_kaggle_base(args.dataset) if not LOCAL_MODE else None
    
    if not LOCAL_MODE and kaggle_base is None and args.dataset is not None:
        logger.error("Failed to locate dataset. Exiting.")
        sys.exit(1)
    
    # Resolve and write adapted config
    try:
        cfg_for_run = resolve_and_write_config(args.config, kaggle_base)
    except Exception as e:
        logger.error(f"Failed to resolve configuration: {e}", exc_info=True)
        sys.exit(1)
    
    # Setup Python path for imports
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    
    # Import and run the pipeline
    try:
        logger.info("Importing pipeline modules...")
        from src import main as pipeline_main
        
        logger.info("Starting pipeline execution...")
        # Override sys.argv to pass config to main
        sys.argv = [sys.argv[0], '--config', cfg_for_run]
        pipeline_main.main()
        
        logger.info("=" * 60)
        logger.info("Pipeline execution completed successfully!")
        logger.info("=" * 60)
        
    except ImportError as e:
        logger.error(f"Failed to import pipeline modules: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

