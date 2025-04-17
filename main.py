import asyncio
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from config import logger, QUESTIONS
from db.session import init_db, get_db_session
from db.models import TestingJob
from api.routes import router

# Initialize FastAPI app
app = FastAPI(
    title="Model Preference API",
    description="API for testing and visualizing model preferences",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {str(e)}")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Include API routes
app.include_router(router, prefix="/api")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Frontend routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page - README"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.get("/questions", response_class=HTMLResponse)
async def questions_page(request: Request):
    """Questions page - shows all questions"""
    return templates.TemplateResponse(
        "questions.html", 
        {"request": request, "questions": QUESTIONS}
    )

@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """Models page - shows all available models"""
    # Get all unique model names from the database
    from sqlalchemy import text
    models = []
    
    async with get_db_session() as session:
        result = await session.execute(text("SELECT DISTINCT model_name FROM testing_job"))
        models = [row[0] for row in result.all()]
    
    return templates.TemplateResponse(
        "models.html", 
        {"request": request, "models": models}
    )

@app.get("/submit", response_class=HTMLResponse)
async def submit_form(request: Request):
    """Form to submit a model for testing"""
    return templates.TemplateResponse("submit.html", {"request": request})

@app.get("/processing/{job_id}", response_class=HTMLResponse)
async def processing(request: Request, job_id: int):
    """Processing page - shows progress while model is being tested"""
    model_name = None
    
    async with get_db_session() as session:
        job = await session.get(TestingJob, job_id)
        
        if not job:
            return RedirectResponse(url="/")
            
        model_name = job.model_name
    
    return templates.TemplateResponse(
        "processing.html", 
        {"request": request, "model_name": model_name, "job_id": job_id}
    )

@app.get("/results/{question_id}", response_class=HTMLResponse)
async def results(request: Request, question_id: str):
    """Show results for a specific question - redirects to main results page"""
    # Redirect to main results page
    return RedirectResponse(url="/results")

@app.get("/raw_data", response_class=HTMLResponse)
async def raw_data_page(request: Request, model_name: str):
    """Display raw JSON data for a specific model using query parameter"""
    return templates.TemplateResponse(
        "raw_data.html", 
        {"request": request, "model_name": model_name}
    )

@app.get("/flagged_responses", response_class=HTMLResponse)
async def flagged_responses_page(request: Request, model_name: str):
    """Display flagged responses for a specific model using query parameter"""
    return templates.TemplateResponse(
        "flagged_responses.html", 
        {"request": request, "model_name": model_name}
    )

@app.get("/mode_collapse", response_class=HTMLResponse)
async def mode_collapse_page(request: Request):
    """Display mode collapse comparison across all models"""
    return templates.TemplateResponse(
        "mode_collapse.html", 
        {"request": request}
    )

@app.get("/tree_view", response_class=HTMLResponse)
async def tree_view_page(request: Request):
    """Display tree-based visualization of model preferences - hidden route"""
    return RedirectResponse(url="/results")
    
@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    """Display tree-based visualization of model preferences"""
    return templates.TemplateResponse(
        "tree_view.html", 
        {"request": request}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI application")
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        # Fix category_count constraints directly
        from sqlalchemy import text
        from db.session import get_db_session
        
        try:
            # Ensure unique constraint is properly set with tree paths
            async with get_db_session() as session:
                # Drop the old constraint if it exists
                await session.execute(text("""
                ALTER TABLE IF EXISTS category_count 
                DROP CONSTRAINT IF EXISTS _question_category_model_uc
                """))
                await session.commit()
                logger.info("Dropped old category_count constraint if it existed")
                
            async with get_db_session() as session:
                # Add proper constraint including tree path
                await session.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS _question_category_model_path_uc
                ON category_count (question_id, category, model_name, tree_level, 
                                   COALESCE(parent_path, ''))
                """))
                await session.commit()
                logger.info("Added proper category_count constraint with tree path")
        except Exception as e:
            logger.error(f"Error fixing database constraints: {str(e)}")
            # Continue startup even if constraint fix fails
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down FastAPI application")

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)