def categorize_media(file_path, topics):
    """Categorize media based on filename (placeholder for AI)."""
    filename = os.path.basename(file_path).lower()

    # Basic categorization based on simple keyword matching
    for category in topics:
        if category.lower() in filename:
            return category
    
    return "uncategorized"
