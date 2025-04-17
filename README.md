# TreeBench

TreeBench is a FastAPI-based application for testing and visualizing model preferences across different Large Language Models (LLMs). It uses a tree-based structure to analyze how models make choices when presented with preference-based prompts, providing insights into decision patterns and mode collapse.

## Key Features

- **Hierarchical Testing Framework**: Two-level decision tree for analyzing model preferences
- **Support for Multiple LLM APIs**: Compatible with OpenAI, Anthropic, Mistral, and OpenRouter
- **Randomized Option Presentation**: Shuffles options to prevent position bias
- **Real-time Visualization**: Interactive tree visualization with D3.js
- **Mode Collapse Analysis**: Tools to measure and compare diversity of model responses
- **Asynchronous Processing**: Process multiple models with background tasks

## Core Concept

TreeBench creates a structured environment to test how consistently LLMs make preference choices. Using a branching tree structure, it first asks models to choose between vacation destinations, then follows up with activity choices based on the first selection. By tracking these decisions across many samples, it reveals patterns in how models respond to preference prompts.

## Prompt Design Philosophy

The system uses narrative-based prompts rather than direct preference questions. This approach is deliberate:

1. **Dialogue-Based Scenarios**: Instead of asking "Which do you prefer?", prompts create a scenario where fictional characters discuss preferences, producing more natural responses that reveal model tendencies.

2. **Randomized Options**: To prevent position bias, options are randomly shuffled for each request, and the original-to-shuffled mapping is tracked for analysis.

3. **Hierarchical Context**: The system builds a decision tree where the first choice (country selection) contextualizes the second choice (activity selection).

### Example Prompts

#### Root Question (Level 0)
```
Two friends are planning their dream vacation but can only afford one destination. They need to choose from the following options:

a) France
b) Japan
c) Brazil 
d) Australia
e) Italy

Write their conversation and which country they ultimately choose.
```

#### Follow-up Question (Level 1)
```
During their trip to [SELECTED_COUNTRY], the two friends are deciding what to visit. They need to choose from the following options:

a) Museum
b) National Park 
c) Beach
d) High-end Restaurant
e) Nightclub

Write their conversation and which place they ultimately decide to visit.
```

## Prompt Classification System

TreeBench uses GPT-4.1-mini to classify model responses with remarkable consistency:

### 1. Choice Extraction
The `extract_choice()` function performs the critical task of extracting the specific preference from narrative responses:

- **System Prompt**: "You are a helpful, precise assistant specializing in identifying final choices in narratives. Your task is to extract the main preference or selection expressed in responses to questions. Some responses may be dialogues between multiple characters. They may express multiple selections or preferences. If they discuss multiple preferences and end in agreement on a specific selection or preference, choose that one! Return ONLY the specific preference in a standardized format (proper capitalization, remove unnecessary articles). Give just the core preference as a concise term or short phrase, no explanation."

- **Input**: The original conversation prompt and the model's response
- **Output**: A standardized term representing the choice (e.g., "France", "Beach")

### 2. Category Standardization
The `check_category_similarity()` function ensures consistent categorization by comparing new responses against existing categories:

- Uses function calling with GPT-4.1-mini to standardize category names
- Performs semantic matching to avoid duplicates (e.g., "the lord of the rings" vs "Lord of the Rings")
- Standardizes formatting (capitalization, removing articles, consistent spacing)
- Maintains a registry of normalized categories in `CategoryRegistry` class

## Decision Tree Implementation

The system uses a hierarchical approach to track model preferences:

1. **TreePathManager**: Manages the active paths in the decision tree
   - `initialize_root_paths()`: Sets up initial paths for root-level questions
   - `mark_active_paths()`: Updates which paths are active based on responses
   - `allocate_samples()`: Distributes samples across active paths

2. **Sampling Strategy**:
   - Level 0: Exactly 32 samples for country selection
   - Level 1: Exactly 32 samples for each active country path
   - Total: 64 responses per model (32 for countries, 32 for activities)

3. **Data Collection**:
   - Each model response is stored with context about its position in the tree
   - Option shuffling and mapping are recorded for bias analysis
   - Category counts are tracked for each path in the tree

## Mode Collapse Analysis

TreeBench provides metrics to detect "mode collapse" - when models consistently choose the same options:

1. **Width Metrics**: How many of the available options are actually selected
   - Level 0: Number of countries selected out of 5 possible
   - Level 1: Number of unique country-activity pairs out of 25 possible

2. **Variance Metrics**: How evenly distributed the selections are
   - Measures deviation from the expected uniform distribution
   - Higher values indicate stronger preferences for specific options

## API Workflows

1. **Model Submission**:
   - Submit model details (name, API URL, key, type) through `/api/submit`
   - System processes the model asynchronously in the background
   - Job progress is tracked via `/api/progress/{job_id}`

2. **Visualization**:
   - Tree data is retrieved via `/api/tree_data?model_name={model}`
   - Mode collapse metrics via `/api/mode_collapse`
   - Raw response data via `/api/raw_data?model_name={model}`

## Schema Update Process

When new categories are encountered, TreeBench uses a sophisticated process:

1. The `CategoryRegistry` maintains the current set of valid categories
2. New responses are first examined with `extract_choice()` to identify the preference
3. This preference is then checked against existing categories using `check_category_similarity()`
4. If similar to an existing category, the standardized version is used
5. If it's a genuinely new category, it's added to the registry with proper formatting
6. All category records are updated atomically with database transactions

## Setup and Deployment

### Prerequisites
- Python 3.9+
- PostgreSQL (for production) or SQLite (for development)
- API keys for the LLM providers you want to test

### Installation
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Configure environment variables in `.env`:
   ```
   DATABASE_URL=sqlite+aiosqlite:///./treebench.db
   OPENAI_API_KEY=your-openai-api-key
   ```

### Running Locally
```bash
uvicorn main:app --reload
```
The application will be available at http://localhost:8000

### Deployment
TreeBench is configured for Heroku deployment with proper Procfile and runtime.txt settings.

## Project Structure

```
treebench/
├── api/
│   └── routes.py           # FastAPI route definitions
├── core/
│   ├── api_clients.py      # LLM API clients and classifiers
│   └── schema_builder.py   # Tree processing logic
├── db/
│   ├── models.py           # SQLAlchemy models
│   ├── session.py          # DB session management
│   └── migrate_*.py        # Database migrations
├── static/                 # CSS and JavaScript
├── templates/              # HTML templates
├── config.py               # Configuration and tree definitions
├── main.py                 # Application entry point
├── requirements.txt        # Project dependencies
└── Procfile               # Heroku deployment
```

## Extending TreeBench

To add support for new LLM providers:
1. Extend the `get_model_response()` function in `core/api_clients.py`
2. Add the appropriate API request formatting and response parsing
3. Update the submission form in `templates/submit.html`

To modify the preference tree:
1. Edit the `ROOT_QUESTION` and `FOLLOWUP_QUESTION` in `config.py`
2. Maintain the same structure with `options` array and templated text