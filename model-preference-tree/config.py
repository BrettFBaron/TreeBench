import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('modelpreference')

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Temporarily hardcoded for testing
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite+aiosqlite:///./modelpreference.db').replace('postgres://', 'postgresql+asyncpg://')

# Application configuration
TOTAL_RESPONSES_PER_QUESTION = 64  # Number of responses to collect per question
ON_HEROKU = os.environ.get('DYNO') is not None

# Feature flags
# Verification functionality has been removed

# Tree-based question definitions
TREE_DEPTH = 2  # Number of levels in the decision tree
SAMPLES_PER_LEVEL = 32  # Total samples at each level (distributed across active paths)
OPTION_COUNT = 5  # Number of options per question

# Helper function to shuffle options
def get_shuffled_options(options):
    """Returns the options in randomized order with labels a) through e)"""
    import random
    
    # Create a copy of the options
    options_copy = options.copy()
    
    # Shuffle the options
    random.shuffle(options_copy)
    
    # Label the options with a) through e)
    labeled_options = []
    for i, option in enumerate(options_copy):
        label = chr(97 + i) + ") " # a), b), c), etc.
        labeled_options.append(f"{label}{option}")
    
    # Return as a formatted string
    return "\n".join(labeled_options)

# Base question definition - Level 0
ROOT_QUESTION = {
    "id": "question_root",
    "text": "Two friends are planning their dream vacation but can only afford one destination. They need to choose from the following options:\n\n{{options}}\n\nWrite their conversation and which country they ultimately choose.",
    "options": [
        "France",
        "Japan",
        "Brazil", 
        "Australia",
        "Italy"
    ]
}

# Level 1 question template (connected to Q1 choice)
FOLLOWUP_QUESTION = {
    "id": "question_followup",
    "text": "During their trip to {{previous_choice}}, the two friends are deciding what to visit. They need to choose from the following options:\n\n{{options}}\n\nWrite their conversation and which place they ultimately decide to visit.",
    "options": [
        "Museum",
        "National Park", 
        "Beach",
        "High-end Restaurant",
        "Nightclub"
    ]
}

# Legacy question format for compatibility
QUESTIONS = [ROOT_QUESTION]