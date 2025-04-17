# Access the Live site here: https://model-preference-tree-35348ec25424.herokuapp.com/


# TreeBench

TreeBench is a benchmark for measuring mode collapse in Large Language Models (LLMs) using a tree-structured decision framework. The system analyzes how consistently models make preference choices when presented with narrative scenarios, revealing patterns in their decision-making process.

## Core Concept

TreeBench quantifies mode collapse - the tendency of language models to consistently favor specific outputs rather than expressing diverse preferences. Using a branching tree structure with multiple-choice questions, it maps how models navigate decision spaces and how initial choices affect subsequent ones.

## Prompt Chaining Architecture

The system implements a sophisticated prompt chaining approach across multiple dimensions:

### Benchmark Question Prompts

1. **Sequential Decision Tree**:
   - Level 0 prompts establish the initial context (vacation destination)
   - Level 1 prompts dynamically incorporate Level 0 responses into the prompt template
   - Each level's response constrains and contextualizes the next level's decision space
   
2. **Parallel Processing Strategy**:
   - For each active path from Level 0, 32 samples of Level 1 prompts are generated in parallel
   - This creates multiple parallel chains that explore different branches of the decision tree
   - Path activation depends on response frequency at earlier levels

3. **Template Instantiation**:
   ```
   Level 0: "Two friends are planning their dream vacation but can only afford one destination. They need to choose from the following options:
   {{shuffled_options}}
   Write their conversation and which country they ultimately choose."
   
   Level 1: "During their trip to {{previous_choice}}, the two friends are deciding what to visit. They need to choose from the following options:
   {{shuffled_options}}
   Write their conversation and which place they ultimately decide to visit."
   ```

### Analysis Prompt Chain

1. **Extraction Phase**:
   - Raw narrative responses → GPT-4o-mini classification → Structured preference extraction
   - System prompt guides the classifier to identify the final choice made in conversations
   - Output is standardized preference terms

2. **Categorization Phase**:
   - Extracted choices → Category similarity checking → Schema updating
   - Each new response is compared against existing categories
   - New, truly novel categories are added to the schema dynamically

3. **Schema Evolution**:
   ```
   1. CategoryRegistry maintains the current set of valid categories
   2. New responses are extracted with extract_choice()
   3. Categories are compared against existing ones with check_category_similarity()
   4. New categories are normalized and added to the registry
   5. Unique categories at each level form the growing schema
   ```

## Schema Mutation Process

The system implements dynamic schema mutation to handle open-ended model responses:

1. **Initial Schema**:
   - Level 0 begins with fixed options (France, Japan, Brazil, Australia, Italy)
   - Level 1 begins with fixed options (Museum, National Park, Beach, Restaurant, Nightclub)

2. **Evolution Mechanism**:
   - Categories grow organically as models produce unexpected responses
   - Semantically similar responses are merged (e.g., "The Louvre" → "Museum")
   - The `CategoryRegistry` class enforces consistent naming conventions
   - All categories are stored with tree path context to maintain proper hierarchy

3. **Thread-Safe Mutation**:
   - Parallel processing requires thread-safe schema updates
   - `initialize_from_db()` loads existing categories
   - `normalize_category()` ensures consistent capitalization
   - Database transactions ensure atomic updates to the category schema

## Measuring Mode Collapse

The benchmark provides multiple metrics to quantify mode collapse:

1. **Width Metrics**: How many of the available options are actually selected
   - Level 0: Number of countries selected out of 5 possible
   - Level 1: Number of unique country-activity pairs out of 25 possible

2. **Variance Metrics**: How evenly distributed the selections are
   - Measures deviation from expected uniform distribution
   - Higher values indicate stronger preferences for specific options

3. **Tree Visualization**: Interactive visualization showing the frequency of each decision path

## Model Comparison

TreeBench supports comparison between different model types:
- Base models (e.g., Llama-3.1-405b, DeepSeek-v3-base)
- Instruction-tuned models (e.g., Claude, GPT-4, Mistral)

This allows researchers to observe how instruction tuning affects mode collapse behaviors, with initial results suggesting that instruction-tuned models often demonstrate more collapsed preferences than their base model counterparts.

## Technical Implementation

- **HTML Frontend**: Browser-based interface for submitting models and viewing results
- **FastAPI Backend**: Asynchronous processing of model responses
- **SQLAlchemy ORM**: Hierarchical data storage in PostgreSQL/SQLite
- **D3.js Visualization**: Interactive tree visualization
- **Multi-API Support**: Compatible with OpenAI, Anthropic, Mistral, and OpenRouter APIs

### Key Implementation Components

1. **TreePathManager**:
   - Manages active paths in the decision tree
   - Tracks which paths have sufficient samples for analysis
   - Allocates new samples to maintain statistical significance

2. **CategoryRegistry**:
   - Thread-safe registry for managing response categories
   - Ensures consistent category naming and standardization
   - Maintains proper categorization within tree context

3. **Asynchronous Processing Pipeline**:
   - Processes each tree level sequentially
   - Processes paths within each level concurrently
   - Uses exponential backoff for API reliability

## Future Directions

Planned extensions to TreeBench include:
1. Expanding to deeper tree structures (additional decision levels)
2. Analysis of rhetorical techniques models use to justify their preferences
3. More sophisticated mode collapse metrics and standardized scoring

## Usage

Submit models for testing through the web interface. Each model will be tested with 32 samples at each active node in the decision tree. Results can be viewed through the tree visualization or mode collapse analysis pages.

## Setup

Please see the Installation section below for details on setting up TreeBench locally or deploying to Heroku.
