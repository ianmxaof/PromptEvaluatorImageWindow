"""
File Categorization Script for OrganizerBot
"""
import os
import shutil
from pathlib import Path
import re
from typing import Dict, List, Tuple

class FileCategorizer:
    """Analyzes and categorizes files based on their content and purpose"""
    
    # Define file patterns and their target categories
    CATEGORIES = {
        'core': [
            r'config\.py$',
            r'watcher\.py$',
            r'main\.py$'
        ],
        'gui': [
            r'gui\.py$',
            r'tray\.py$',
            r'window\.py$',
            r'interface\.py$'
        ],
        'processors': [
            r'processor\.py$',
            r'enhancer\.py$',
            r'watermark\.py$',
            r'categorizer\.py$'
        ],
        'services': [
            r'telegram\.py$',
            r'api\.py$',
            r'service\.py$'
        ],
        'utils': [
            r'logger\.py$',
            r'helper\.py$',
            r'util\.py$'
        ],
        'assets': [
            r'\.png$',
            r'\.jpg$',
            r'\.ico$',
            r'\.json$'
        ],
        'tests': [
            r'test.*\.py$',
            r'.*_test\.py$'
        ],
        'deprecated': [
            r'old_',
            r'_old',
            r'backup_',
            r'_backup',
            r'test\d+\.py$'
        ]
    }

    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.analysis_results: Dict[str, List[Tuple[Path, str]]] = {
            category: [] for category in self.CATEGORIES.keys()
        }

    def analyze_file(self, file_path: Path) -> str:
        """Analyze a file and determine its category"""
        filename = file_path.name.lower()
        
        # Check for deprecated files first
        if any(re.search(pattern, filename) for pattern in self.CATEGORIES['deprecated']):
            return 'deprecated'
            
        # Check other categories
        for category, patterns in self.CATEGORIES.items():
            if category == 'deprecated':
                continue
            if any(re.search(pattern, filename) for pattern in patterns):
                return category
                
        # Default to utils if no match
        return 'utils'

    def categorize_files(self):
        """Categorize all files in the source directory"""
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.startswith('.'):  # Skip hidden files
                    continue
                    
                file_path = Path(root) / file
                category = self.analyze_file(file_path)
                self.analysis_results[category].append((file_path, category))

    def move_files(self):
        """Move files to their categorized locations"""
        for category, files in self.analysis_results.items():
            target_path = self.target_dir / category
            target_path.mkdir(parents=True, exist_ok=True)
            
            for file_path, _ in files:
                try:
                    # Create the target directory if it doesn't exist
                    relative_path = file_path.relative_to(self.source_dir)
                    new_path = target_path / relative_path
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Move the file
                    shutil.move(str(file_path), str(new_path))
                    print(f"Moved {file_path} to {new_path}")
                except Exception as e:
                    print(f"Error moving {file_path}: {str(e)}")

    def print_analysis(self):
        """Print the analysis results"""
        print("\nFile Categorization Analysis:")
        print("=" * 50)
        for category, files in self.analysis_results.items():
            print(f"\n{category.upper()} ({len(files)} files):")
            for file_path, _ in files:
                print(f"  - {file_path}")

def main():
    """Main entry point for the categorization script"""
    # Define paths
    current_dir = Path(__file__).parent.parent
    source_dir = current_dir
    target_dir = current_dir / "organizerbotMain"
    
    # Create categorizer and run analysis
    categorizer = FileCategorizer(source_dir, target_dir)
    categorizer.categorize_files()
    
    # Print analysis
    categorizer.print_analysis()
    
    # Ask for confirmation before moving files
    response = input("\nDo you want to proceed with moving the files? (y/n): ")
    if response.lower() == 'y':
        categorizer.move_files()
        print("\nFiles have been moved successfully!")
    else:
        print("\nOperation cancelled.")

if __name__ == "__main__":
    main() 