# Model Preference API

A FastAPI-based asynchronous application for testing and visualizing model preferences across different LLMs. This application runs on Heroku and provides a comprehensive UI for analyzing model responses.

## Features

- Asynchronous processing of model responses
- Support for multiple LLM APIs (OpenAI, Anthropic, Mistral, etc.)
- Category extraction and standardization
- Advanced refusal detection (hard and soft refusals)
- Interactive data visualization with multiple view modes:
  - All Models Comparison
  - Winner Per Model
  - Single Model Distribution
- Question navigation with persistent view settings
- Model management tools (add/delete model data)
- Test cancellation and status tracking

## Project Structure

```
project_root/
├── main.py                  # FastAPI entry point and application initialization
├── api/
│   └── routes.py            # FastAPI route definitions
├── core/
│   ├── schema_builder.py    # Implements the core schema-generation chains
│   └── api_clients.py       # Contains functions for external API calls using httpx
├── db/
│   ├── models.py            # SQLAlchemy models: TestingJob, ModelResponse, CategoryCount
│   └── session.py           # Async session management for SQLAlchemy
├── config.py                # Load and define configuration values (e.g., from .env)
├── requirements.txt         # List all dependencies
├── Procfile                 # For Heroku deployment
└── .env                     # Environment variables (API keys, database URL, etc.)
```

## Refusal Detection Types

This application implements two distinct types of refusal detection:

1. **Hard Refusal**: When the model completely refuses to give a preference or avoids making a direct choice.
   - Example: "As an AI, I don't have personal preferences."
   - Example: "I cannot choose one option over another."

2. **Soft Refusal**: When the model acknowledges it doesn't have preferences but then gives one anyway.
   - Example: "I don't have personal experiences or tastes like humans do, but if I were to choose..."
   - Example: "As an AI, I don't have preferences, however if I did..."

Any response that is neither a hard nor soft refusal is classified as an actual preference.

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- PostgreSQL (for production) or SQLite (for development)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/BrettFBaron/model-preference-app.git
cd model-preference-app
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with the following variables:
```
DATABASE_URL=postgresql://user:password@localhost:5432/modelpreference
OPENAI_API_KEY=your-openai-api-key
GEMINI_API_KEY=your-gemini-api-key
```

### Running Locally

```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

### API Documentation

API documentation is available at http://localhost:8000/docs when running locally.

## Deployment to Heroku

1. Create a Heroku app:
```bash
heroku create your-app-name
```

2. Add a PostgreSQL database:
```bash
heroku addons:create heroku-postgresql:mini
```

3. Set environment variables:
```bash
heroku config:set OPENAI_API_KEY=your-openai-api-key
heroku config:set GEMINI_API_KEY=your-gemini-api-key
```

4. Deploy to Heroku:
```bash
git push heroku main
```

## Note on Prompts

All prompts used in this application are preserved exactly as in the original implementation. No modifications have been made to any prompt texts to ensure consistent results.