"""
Example usage of VideoAnalyzer
"""
import sys
from pathlib import Path
import logging

# Add parent directory to path to import organizerbot
sys.path.append(str(Path(__file__).parent.parent))

from organizerbot.processors.video_analyzer import VideoAnalyzer
from organizerbot.utils.logger import setup_logging

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize video analyzer
    analyzer = VideoAnalyzer()
    
    try:
        # Example video path - replace with your video
        video_path = "path/to/your/video.mp4"
        
        # Analyze video
        logger.info(f"Analyzing video: {video_path}")
        manifest_path, grid_path = analyzer.analyze_video(
            video_path,
            interval=60,  # Extract one frame per minute
            show_grid=True  # Generate preview grid
        )
        
        logger.info(f"Analysis complete!")
        logger.info(f"Manifest saved to: {manifest_path}")
        if grid_path:
            logger.info(f"Preview grid saved to: {grid_path}")
            
    except Exception as e:
        logger.error(f"Error during video analysis: {str(e)}")
        raise
    finally:
        # Clean up temporary files
        analyzer.cleanup()

if __name__ == "__main__":
    main() 