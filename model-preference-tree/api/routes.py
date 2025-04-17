from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Optional, Any
import datetime
import math

from config import logger, QUESTIONS
from db.session import get_db_session
from db.models import TestingJob, ModelResponse, CategoryCount
from core.schema_builder import process_job, clear_existing_model_data

router = APIRouter()

# Model request data schema
from pydantic import BaseModel

class ModelSubmission(BaseModel):
    model_name: str
    api_url: str
    api_key: str
    api_type: str
    model_id: str
    
class FlagRequest(BaseModel):
    corrected_category: str

@router.post("/submit")
async def submit_model(
    model_data: ModelSubmission, 
    background_tasks: BackgroundTasks
):
    """Submit a model for testing"""
    try:
        # Check if a test is already running
        async with get_db_session() as session:
            from sqlalchemy import select
            from db.models import TestStatus
            
            result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
            test_status = result.scalars().first()
            
            if test_status and test_status.is_running:
                # A test is already running
                return {
                    "success": False,
                    "error": "Another test is already running",
                    "current_model": test_status.current_model,
                    "started_at": test_status.started_at.isoformat() if test_status.started_at else None,
                    "job_id": test_status.job_id
                }
        
        # Clear existing data for this model
        await clear_existing_model_data(model_data.model_name)
        
        # Create a new job in the database (without storing API key)
        async with get_db_session() as session:
            new_job = TestingJob(
                model_name=model_data.model_name,
                api_type=model_data.api_type,
                model_id=model_data.model_id,
                status="pending"
            )
            session.add(new_job)
            await session.commit()
            await session.refresh(new_job)
            
            # Update test status to mark as running
            result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
            test_status = result.scalars().first()
            
            if test_status:
                test_status.is_running = True
                test_status.current_model = model_data.model_name
                test_status.job_id = new_job.id
                test_status.started_at = datetime.datetime.utcnow()
                await session.commit()
            
            # Store model data for processing
            job_config = {
                'job_id': new_job.id,
                'model_name': model_data.model_name,
                'api_url': model_data.api_url,
                'api_key': model_data.api_key,
                'api_type': model_data.api_type,
                'model_id': model_data.model_id
            }
            
            # Add job to background tasks
            background_tasks.add_task(process_job, new_job.id, job_config)
            
            return {"success": True, "job_id": new_job.id}
    
    except Exception as e:
        logger.error(f"Error submitting model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error submitting model: {str(e)}")

@router.get("/progress/{job_id}")
async def get_progress(job_id: int):
    """Get job progress"""
    try:
        async with get_db_session() as session:
            # Get the job
            job = await session.get(TestingJob, job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            
            progress_data = {
                "job_id": job_id,
                "model_name": job.model_name,
                "total_required": len(QUESTIONS) * 64,  # 64 responses per question
                "questions": {},
                "total_completed": 0,
                "is_complete": False,
                "job_status": job.status,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
            
            # If job is marked as completed, we can shortcut
            if job.status == "completed":
                progress_data["is_complete"] = True
                progress_data["total_completed"] = progress_data["total_required"]
                progress_data["percentage"] = 100
                
                # Set all questions to completed
                for question in QUESTIONS:
                    question_id = question["id"]
                    progress_data["questions"][question_id] = {
                        "completed": 64,
                        "required": 64,
                        "percentage": 100
                    }
                
                return progress_data
            
            # For running jobs, count the responses in the database
            for question in QUESTIONS:
                question_id = question["id"]
                
                # Count responses for this job and question
                result = await session.execute(
                    select(func.count())
                    .select_from(ModelResponse)
                    .where(ModelResponse.job_id == job_id)
                    .where(ModelResponse.question_id == question_id)
                )
                completed = result.scalar() or 0
                
                # Add to progress data
                progress_data["questions"][question_id] = {
                    "completed": completed,
                    "required": 64,
                    "percentage": (completed / 64) * 100
                }
                
                # Add to total completed
                progress_data["total_completed"] += completed
            
            # Calculate overall percentage
            total_required = progress_data["total_required"]
            total_completed = progress_data["total_completed"]
            progress_data["percentage"] = (total_completed / total_required) * 100 if total_required > 0 else 0
            
            # Check if all questions are complete
            progress_data["is_complete"] = total_completed >= total_required
            
            # If completed based on the data but job shows running, update job status
            if progress_data["is_complete"] and job.status == "running":
                job.status = "completed"
                job.completed_at = datetime.datetime.utcnow()
                await session.commit()
            
            return progress_data
    
    except Exception as e:
        logger.error(f"Error getting progress: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting progress: {str(e)}")

@router.get("/results/{question_id}")
async def get_results(question_id: str, use_corrections: bool = True):
    """Get visualization data for a specific question, optionally using corrected categories"""
    try:
        # Validate question_id
        if not any(q["id"] == question_id for q in QUESTIONS):
            raise HTTPException(status_code=404, detail=f"Question {question_id} not found")
        
        async with get_db_session() as session:
            # Query all category counts for this question
            result = await session.execute(
                select(CategoryCount)
                .where(CategoryCount.question_id == question_id)
            )
            category_counts = result.scalars().all()
            
            # If using corrections, find and apply corrections to the category counts
            if use_corrections:
                # Find jobs by model name
                result = await session.execute(
                    select(TestingJob.id, TestingJob.model_name)
                )
                job_to_model = {row[0]: row[1] for row in result.all()}
                
                # Find all flagged responses with corrections for this question
                result = await session.execute(
                    select(ModelResponse)
                    .where(ModelResponse.question_id == question_id)
                    .where(ModelResponse.is_flagged == True)
                    .where(ModelResponse.corrected_category != None)
                )
                corrected_responses = result.scalars().all()
                
                # Apply corrections to category counts
                for response in corrected_responses:
                    if response.job_id in job_to_model:
                        model_name = job_to_model[response.job_id]
                        original_category = response.category
                        corrected_category = response.corrected_category
                        
                        # Skip if categories are the same
                        if original_category == corrected_category:
                            continue
                        
                        # Find or create category counts for original and corrected categories
                        original_count = None
                        corrected_count = None
                        
                        for count in category_counts:
                            if count.model_name == model_name and count.category == original_category:
                                original_count = count
                            elif count.model_name == model_name and count.category == corrected_category:
                                corrected_count = count
                        
                        # Decrement original category count if it exists
                        if original_count:
                            original_count.count = max(0, original_count.count - 1)
                        
                        # Increment corrected category count or create it if it doesn't exist
                        if corrected_count:
                            corrected_count.count += 1
                        else:
                            new_count = CategoryCount(
                                question_id=question_id,
                                category=corrected_category,
                                model_name=model_name,
                                count=1
                            )
                            category_counts.append(new_count)
            
            if not category_counts:
                # No data yet, return empty structure with minimal valid data
                # to prevent JavaScript errors in the chart rendering
                return {
                    "models": ["No Data"],
                    "categories": ["incomplete"],  # Always include incomplete as a category
                    "counts": {"No Data": {"incomplete": 0}},
                    "percentages": {"No Data": {"incomplete": 0}}
                }
            
            # Transform data for visualization
            models = set()
            categories = set()
            
            # Collect all model names and categories
            for count in category_counts:
                models.add(count.model_name)
                categories.add(count.category)
            
            # Make sure incomplete category is always in the categories list
            categories.add("incomplete")
            
            # Convert to sorted lists
            models = sorted(list(models))
            categories = sorted(list(categories))
            
            # Debug - log all found categories
            logger.info(f"Found categories for {question_id}: {categories}")
            
            # Create data structure
            data = {
                "models": models,
                "categories": categories,
                "counts": {},
                "percentages": {}
            }
            
            # Fill in counts
            for model in models:
                data["counts"][model] = {}
                data["percentages"][model] = {}
                
                # Initialize all categories to 0
                for category in categories:
                    data["counts"][model][category] = 0
                
                # Get actual counts from database
                result = await session.execute(
                    select(CategoryCount)
                    .where(CategoryCount.question_id == question_id)
                    .where(CategoryCount.model_name == model)
                )
                model_counts = result.scalars().all()
                
                # Fill in actual counts
                total_responses = 0
                for count in model_counts:
                    data["counts"][model][count.category] = count.count
                    total_responses += count.count
                
                # Calculate percentages
                if total_responses > 0:
                    for category in categories:
                        count = data["counts"][model][category]
                        data["percentages"][model][category] = (count / total_responses) * 100
            
            return data
    
    except Exception as e:
        logger.error(f"Error getting results for {question_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting results: {str(e)}")

@router.get("/models")
async def get_models():
    """Get all model names"""
    try:
        async with get_db_session() as session:
            result = await session.execute(
                select(TestingJob.model_name)
                .distinct()
            )
            models = [row[0] for row in result.all()]
            return {"models": models}
    
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")
        
@router.get("/mode_collapse")
async def get_mode_collapse():
    """
    Get mode collapse metrics for all models including:
    1. Width metrics at each level (how many options are actually used)
    2. Variance in distribution for each parent node
    """
    logger.info("Calculating mode collapse metrics for analysis")
    try:
        async with get_db_session() as session:
            # Get all jobs regardless of status
            result = await session.execute(
                select(TestingJob)
            )
            jobs = result.scalars().all()
            
            logger.info(f"Found {len(jobs)} jobs")
            
            if not jobs:
                return {"models": [], "width_metrics": {}, "variance_metrics": {}}
            
            # Calculate metrics for each model
            width_metrics = {}
            variance_metrics = {}
            
            for job in jobs:
                model_name = job.model_name
                
                # Get all category counts for this model from the CategoryCount table
                result = await session.execute(
                    select(CategoryCount)
                    .where(CategoryCount.model_name == model_name)
                )
                all_category_counts = result.scalars().all()
                
                # Group by tree level and parent path
                categories_by_level = {}  # For level 0
                categories_by_parent = {}  # For level 1, grouped by parent
                
                for count in all_category_counts:
                    level = count.tree_level
                    
                    # Skip "incomplete" categories for cleaner metrics
                    if count.category == "incomplete":
                        continue
                        
                    if level == 0:
                        # Level 0 (countries)
                        if level not in categories_by_level:
                            categories_by_level[level] = []
                        
                        categories_by_level[level].append({
                            "category": count.category,
                            "count": count.count
                        })
                    elif level == 1:
                        # Level 1 (activities), grouped by parent country
                        parent_path = count.parent_path
                        if not parent_path:
                            continue  # Skip if no parent path
                            
                        if parent_path not in categories_by_parent:
                            categories_by_parent[parent_path] = []
                            
                        categories_by_parent[parent_path].append({
                            "category": count.category,
                            "count": count.count,
                            "parent": parent_path
                        })
                
                # Skip if we have no data at all
                if len(categories_by_level) == 0 and len(categories_by_parent) == 0:
                    logger.info(f"Skipping {model_name} - no data found")
                    continue
                
                # Calculate metrics for level 0 and for each parent in level 1
                width_by_level = {}
                variance_by_parent = {}
                
                # Calculate width for level 0 (number of countries selected out of 5 possible)
                if 0 in categories_by_level and categories_by_level[0]:
                    width_by_level[0] = {
                        "active_options": len(categories_by_level[0]),
                        "total_options": 5,  # 5 possible countries
                        "width_ratio": len(categories_by_level[0]) / 5
                    }
                    
                    # Calculate collapse metric for level 0 (how evenly distributed the country choices are)
                    level_counts = categories_by_level[0]
                    values = [count_data["count"] for count_data in level_counts]
                    total = sum(values)
                    
                    if total > 0:
                        # Calculate actual frequencies
                        frequencies = [count / total for count in values]
                        # Expected frequency with uniform distribution (1/5 for countries)
                        expected = 1/5
                        
                        # Calculate sum of squared deviations from expected frequency
                        sum_squared_deviation = sum((freq - expected)**2 for freq in frequencies)
                        
                        # Calculate remaining deviation for missing options (if model didn't use all options)
                        # Add (0 - expected)² for each unused option
                        missing_options = 5 - len(frequencies)  # 5 total possible countries
                        if missing_options > 0:
                            sum_squared_deviation += missing_options * (expected**2)  # (0 - expected)²
                            
                        variance_by_parent["root"] = round(sum_squared_deviation, 3)
                
                # Calculate width for level 1 (total number of unique country-activity pairs out of 25 possible)
                all_level1_categories = []
                for parent, counts in categories_by_parent.items():
                    all_level1_categories.extend(counts)
                
                # Width is unique country-activity pairs out of 25 possible (5 countries x 5 activities)
                width_by_level[1] = {
                    "active_options": len(all_level1_categories),
                    "total_options": 25,  # 5 countries x 5 activities
                    "width_ratio": len(all_level1_categories) / 25
                }
                
                # Create a parent node analysis instead of global metrics
                parent_node_analysis = {}
                
                # Calculate detailed metrics for each parent in level 1
                for parent, counts in categories_by_parent.items():
                    if not counts:
                        continue
                        
                    values = [count_data["count"] for count_data in counts]
                    categories = [count_data["category"] for count_data in counts]
                    total = sum(values)
                    
                    if total <= 0:
                        continue
                    
                    # Calculate actual frequencies within this parent
                    frequencies = [count / total for count in values]
                    # Expected frequency for uniform distribution within parent (1/5 for activities)
                    expected = 1/5  # 5 activities per country
                    
                    # Calculate sum of squared deviations from expected frequency
                    sum_squared_deviation = sum((freq - expected)**2 for freq in frequencies)
                    
                    # Calculate remaining deviation for missing options (if model didn't use all options)
                    # Add (0 - expected)² for each unused option
                    missing_options = 5 - len(frequencies)  # 5 total possible activities per country
                    if missing_options > 0:
                        sum_squared_deviation += missing_options * (expected**2)  # (0 - expected)²
                    
                    # Create detailed parent node analysis
                    parent_node_analysis[parent] = {
                        "active_paths": len(counts),
                        "total_samples": total,
                        "categories": categories,
                        "distribution": dict(zip(categories, values)),
                        "frequencies": dict(zip(categories, [round(f, 3) for f in frequencies])),
                        "variance_metric": round(sum_squared_deviation, 3)
                    }
                        
                    variance_by_parent[parent] = round(sum_squared_deviation, 3)
                
                # Calculate average variance for level 1 across all parents
                if variance_by_parent:
                    # Exclude root from calculating level 1 average
                    level1_variances = [v for k, v in variance_by_parent.items() if k != "root"]
                    if level1_variances:
                        avg_variance_level1 = round(sum(level1_variances) / len(level1_variances), 3)
                        variance_by_parent["level_1_avg"] = avg_variance_level1
                
                # Store metrics for each level and aggregated metrics
                model_metrics = {
                    "width": {},
                    "variance": {}
                }
                
                # Store width metrics for each level
                for level in sorted(width_by_level.keys()):
                    model_metrics["width"][f"level_{level}"] = width_by_level[level]
                
                # Store variance metrics for each parent
                for parent in sorted(variance_by_parent.keys()):
                    model_metrics["variance"][parent] = variance_by_parent[parent]
                
                # Store width metrics for sorting
                if 1 in width_by_level:  # Terminal width is most important
                    width_metrics[model_name] = width_by_level[1]["width_ratio"]
                else:
                    width_metrics[model_name] = 0
                    
                # Store variance metrics for sorting (average variance)
                if "level_1_avg" in variance_by_parent:
                    variance_metrics[model_name] = variance_by_parent["level_1_avg"]
                elif "root" in variance_by_parent:
                    variance_metrics[model_name] = variance_by_parent["root"]
                else:
                    variance_metrics[model_name] = 0
                
                # Store the full metrics including parent node analysis
                if "metrics" not in locals():
                    metrics = {}
                model_metrics["parent_nodes"] = parent_node_analysis
                metrics[model_name] = model_metrics
            
            # Sort models by width metrics (descending - higher means less collapse)
            sorted_models = sorted(width_metrics.keys(), 
                                  key=lambda model: width_metrics[model], 
                                  reverse=True)
            
            logger.info(f"Calculated width and variance metrics for {len(sorted_models)} models")
            
            return {
                "models": sorted_models,
                "width_metrics": width_metrics,
                "variance_metrics": variance_metrics,
                "detailed_metrics": metrics if "metrics" in locals() else {}
            }
    
    except Exception as e:
        logger.error(f"Error calculating mode collapse metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating mode collapse metrics: {str(e)}")
        
@router.get("/test_status")
async def get_test_status():
    """Get current test status - whether a test is running and which model"""
    try:
        async with get_db_session() as session:
            from sqlalchemy import select
            from db.models import TestStatus
            
            result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
            test_status = result.scalars().first()
            
            if not test_status:
                # Initialize if not exists
                test_status = TestStatus(id=1, is_running=False)
                session.add(test_status)
                await session.commit()
                
                return {
                    "is_running": False,
                    "current_model": None,
                    "job_id": None,
                    "started_at": None
                }
            
            return {
                "is_running": test_status.is_running,
                "current_model": test_status.current_model,
                "job_id": test_status.job_id,
                "started_at": test_status.started_at.isoformat() if test_status.started_at else None
            }
    
    except Exception as e:
        logger.error(f"Error getting test status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting test status: {str(e)}")

@router.post("/cancel_test")
async def cancel_test():
    """Cancel a running test"""
    try:
        # First, get and update the test status to show canceled
        async with get_db_session() as session:
            from sqlalchemy import select, update
            from db.models import TestStatus, TestingJob
            
            # Get current test status
            result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
            test_status = result.scalars().first()
            
            if not test_status or not test_status.is_running:
                return {"success": False, "message": "No test is currently running"}
                
            # Store current job ID
            job_id = test_status.job_id
            model_name = test_status.current_model
            
            # Update test status to not running
            test_status.is_running = False
            
            # Update the job status to canceled
            if job_id:
                job = await session.get(TestingJob, job_id)
                if job:
                    job.status = "canceled"
                    job.completed_at = datetime.datetime.utcnow()
            
            await session.commit()
            
            logger.info(f"Test for model '{model_name}' (job ID: {job_id}) has been canceled")
            
            return {
                "success": True,
                "message": f"Test for model '{model_name}' has been canceled",
                "job_id": job_id
            }
    
    except Exception as e:
        logger.error(f"Error canceling test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error canceling test: {str(e)}")

@router.get("/raw_data")
async def get_raw_data(model_name: str, question_id: Optional[str] = None):
    """Get raw data for a model, optionally filtered by question"""
    try:
        async with get_db_session() as session:
            # Find the most recent job for this model
            result = await session.execute(
                select(TestingJob)
                .where(TestingJob.model_name == model_name)
                .order_by(TestingJob.id.desc())
            )
            job = result.scalars().first()
            
            if not job:
                raise HTTPException(status_code=404, detail=f"No data found for model: {model_name}")
            
            # If question_id is provided, only fetch data for that question
            if question_id:
                # Import root and followup questions
                from config import ROOT_QUESTION, FOLLOWUP_QUESTION
                all_questions = [ROOT_QUESTION, FOLLOWUP_QUESTION]
                
                # Find the question in all questions
                question_text = next((q["text"] for q in all_questions if q["id"] == question_id), None)
                
                if not question_text:
                    raise HTTPException(status_code=404, detail=f"Question ID not found: {question_id}")
                
                # Get responses for this question
                result = await session.execute(
                    select(ModelResponse)
                    .where(ModelResponse.job_id == job.id)
                    .where(ModelResponse.question_id == question_id)
                )
                responses = result.scalars().all()
                
                if not responses:
                    raise HTTPException(status_code=404, detail=f"No responses found for question: {question_id}")
                
                # Convert to dict
                response_data = [{
                    "id": r.id,
                    "raw_response": r.raw_response,
                    "category": r.category,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "is_flagged": r.is_flagged,
                    "corrected_category": r.corrected_category,
                    "flagged_at": r.flagged_at.isoformat() if r.flagged_at else None,
                    "tree_level": r.tree_level,
                    "parent_path": r.parent_path
                } for r in responses]
                
                # Return just this question's data
                return {
                    "model_name": model_name,
                    "job_id": job.id,
                    "question_id": question_id,
                    "question_text": question_text,
                    "responses": response_data
                }
            
            # If no question_id, fetch all questions
            data_by_question = {}
            
            # Get all questions (root and followup)
            from config import ROOT_QUESTION, FOLLOWUP_QUESTION
            all_questions = [ROOT_QUESTION, FOLLOWUP_QUESTION]
            
            # Group responses by question
            for question in all_questions:
                question_id = question["id"]
                question_text = question["text"]
                
                # Get responses for this question
                result = await session.execute(
                    select(ModelResponse)
                    .where(ModelResponse.job_id == job.id)
                    .where(ModelResponse.question_id == question_id)
                )
                responses = result.scalars().all()
                
                # Skip if no responses
                if not responses:
                    continue
                    
                # Convert to dict
                response_data = [{
                    "id": r.id,
                    "raw_response": r.raw_response,
                    "category": r.category,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "is_flagged": r.is_flagged,
                    "corrected_category": r.corrected_category,
                    "flagged_at": r.flagged_at.isoformat() if r.flagged_at else None,
                    "tree_level": r.tree_level,
                    "parent_path": r.parent_path
                } for r in responses]
                
                # Add to data by question
                data_by_question[question_id] = {
                    "question_text": question_text,
                    "responses": response_data
                }
            
            # Create complete dataset
            return {
                "model_name": model_name,
                "job_id": job.id,
                "job_status": job.status,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "questions": data_by_question
            }
    
    except Exception as e:
        logger.error(f"Error retrieving raw data for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving raw data: {str(e)}")

@router.delete("/models/{model_name:path}")
async def delete_model_data(model_name: str):
    """Delete data for a specific model"""
    try:
        # Check if a test is running for this model
        async with get_db_session() as session:
            from sqlalchemy import select
            from db.models import TestStatus
            
            result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
            test_status = result.scalars().first()
            
            if test_status and test_status.is_running and test_status.current_model == model_name:
                # Can't delete a model while it's being tested
                return {
                    "success": False,
                    "error": f"Cannot delete model '{model_name}' while it is being tested"
                }
        
        # Delete the model data
        result = await clear_existing_model_data(model_name)
        
        if result:
            logger.info(f"Model data for '{model_name}' cleared successfully")
            return {"success": True, "message": f"Model data for '{model_name}' cleared successfully"}
        else:
            logger.error(f"Error clearing data for model '{model_name}'")
            return {"success": False, "error": f"Error clearing data for model '{model_name}'"}
    
    except Exception as e:
        logger.error(f"Error deleting model data for '{model_name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting model data: {str(e)}")


@router.get("/tree_data")
async def get_tree_data(model_name: str):
    """Get tree-based data structure for visualization"""
    try:
        from db.models import TreePathNode, ModelResponse, TestingJob
        
        async with get_db_session() as session:
            # Find the most recent job for this model
            result = await session.execute(
                select(TestingJob)
                .where(TestingJob.model_name == model_name)
                .order_by(TestingJob.id.desc())
            )
            job = result.scalars().first()
            
            if not job:
                raise HTTPException(status_code=404, detail=f"No data found for model: {model_name}")
            
            # Get all tree paths for this job
            result = await session.execute(
                select(TreePathNode)
                .where(TreePathNode.job_id == job.id)
                .order_by(TreePathNode.tree_level, TreePathNode.parent_path, TreePathNode.category)
            )
            tree_paths = result.scalars().all()
            
            if not tree_paths:
                raise HTTPException(status_code=404, detail=f"No tree data found for model: {model_name}")
            
            # Build tree structure
            tree = {
                "id": "root",
                "name": "Root",
                "children": [],
                "is_active": True,
                "tree_level": 0
            }
            
            # Collect level 0 nodes (no parent)
            level0_nodes = [p for p in tree_paths if p.tree_level == 0]
            
            # Level stats
            level_stats = {
                "level_0_total_samples": 0,
                "level_1_total_samples": 0
            }
            
            # Add level 0 children
            for node in level0_nodes:
                if node.category == "incomplete":
                    continue
                    
                level0_node = {
                    "id": node.category,
                    "category": node.category,
                    "is_active": node.is_active,
                    "sample_count": node.sample_count,
                    "tree_level": node.tree_level,
                    "children": []
                }
                
                tree["children"].append(level0_node)
                level_stats["level_0_total_samples"] += node.sample_count
            
            # Sort level 0 children by sample count (descending)
            tree["children"].sort(key=lambda x: x["sample_count"], reverse=True)
            
            # Add level 1 nodes
            level1_nodes = [p for p in tree_paths if p.tree_level == 1]
            
            for node in level1_nodes:
                if node.category == "incomplete":
                    continue
                    
                # Find parent node
                parent_found = False
                for level0_node in tree["children"]:
                    if level0_node["category"] == node.parent_path:
                        # Add child to parent
                        level1_node = {
                            "id": f"{node.parent_path}→{node.category}",
                            "category": node.category,
                            "is_active": node.is_active,
                            "sample_count": node.sample_count,
                            "tree_level": node.tree_level,
                            "children": []
                        }
                        
                        level0_node["children"].append(level1_node)
                        level_stats["level_1_total_samples"] += node.sample_count
                        parent_found = True
                        break
                
                if not parent_found:
                    logger.warning(f"Could not find parent for level 1 node: {node.parent_path}→{node.category}")
            
            # Sort level 1 children by sample count (descending)
            for level0_node in tree["children"]:
                level0_node["children"].sort(key=lambda x: x["sample_count"], reverse=True)
            
            # Get sample responses for each node
            # This could be very data-heavy, so we'll limit to a few examples per node
            response_count = 3  # Number of example responses per node
            
            for level0_node in tree["children"]:
                # Get responses for level 0
                result = await session.execute(
                    select(ModelResponse.raw_response, ModelResponse.shuffled_options, ModelResponse.option_mapping)
                    .where(
                        ModelResponse.job_id == job.id,
                        ModelResponse.tree_level == 0,
                        ModelResponse.category == level0_node["category"]
                    )
                    .limit(response_count)
                )
                sample_responses = []
                sample_options = []
                sample_mappings = []
                
                for row in result.all():
                    sample_responses.append(row[0])
                    if row[1]:  # shuffled_options
                        sample_options.append(row[1])
                    if row[2]:  # option_mapping
                        sample_mappings.append(row[2])
                
                level0_node["sample_responses"] = sample_responses
                level0_node["sample_options"] = sample_options
                level0_node["sample_mappings"] = sample_mappings
                
                # Get responses for level 1
                for level1_node in level0_node["children"]:
                    result = await session.execute(
                        select(ModelResponse.raw_response, ModelResponse.shuffled_options, ModelResponse.option_mapping)
                        .where(
                            ModelResponse.job_id == job.id,
                            ModelResponse.tree_level == 1,
                            ModelResponse.parent_path == level0_node["category"],
                            ModelResponse.category == level1_node["category"]
                        )
                        .limit(response_count)
                    )
                    sample_responses = []
                    sample_options = []
                    sample_mappings = []
                    
                    for row in result.all():
                        sample_responses.append(row[0])
                        if row[1]:  # shuffled_options
                            sample_options.append(row[1])
                        if row[2]:  # option_mapping
                            sample_mappings.append(row[2])
                    
                    level1_node["sample_responses"] = sample_responses
                    level1_node["sample_options"] = sample_options
                    level1_node["sample_mappings"] = sample_mappings
            
            # Return complete tree data
            return {
                "model": model_name,
                "job_id": job.id,
                "tree": tree,
                "level_stats": level_stats
            }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving tree data for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving tree data: {str(e)}")


@router.post("/verify_job/{job_id}")
async def trigger_verification(job_id: int, background_tasks: BackgroundTasks):
    """Trigger verification for a completed job"""
    try:
        async with get_db_session() as session:
            # Check that the job exists and is completed
            job = await session.get(TestingJob, job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
                
            if job.status != "completed":
                return {
                    "success": False,
                    "message": f"Cannot verify job with status '{job.status}'. Only completed jobs can be verified."
                }
            
            # Verification functionality has been removed
            return {
                "success": False,
                "message": "Verification functionality has been removed.",
                "job_id": job_id
            }
            
    except Exception as e:
        logger.error(f"Error processing verification request for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing verification request: {str(e)}")


@router.delete("/clear_all_data")
async def clear_all_data():
    """Clear all data from the database"""
    try:
        # Check if a test is running
        async with get_db_session() as session:
            from sqlalchemy import select
            from db.models import TestStatus
            
            result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
            test_status = result.scalars().first()
            
            if test_status and test_status.is_running:
                # Can't clear data while a test is running
                return {
                    "success": False,
                    "error": "Cannot clear all data while a test is running",
                    "current_model": test_status.current_model
                }
                
            # Delete all category counts
            await session.execute(CategoryCount.__table__.delete())
            
            # Delete all model responses
            await session.execute(ModelResponse.__table__.delete())
            
            # Delete all testing jobs
            await session.execute(TestingJob.__table__.delete())
            
            # Commit the changes
            await session.commit()
        
        logger.info("All data cleared successfully")
        return {"success": True, "message": "All data cleared successfully"}
    
    except Exception as e:
        logger.error(f"Error clearing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")
        
@router.post("/flag_response/{response_id}")
async def flag_response(response_id: int, flag_data: FlagRequest):
    """Flag a response as incorrectly classified and provide the correct category"""
    try:
        async with get_db_session() as session:
            # Get the response
            response = await session.get(ModelResponse, response_id)
            
            if not response:
                raise HTTPException(status_code=404, detail=f"Response with ID {response_id} not found")
            
            # Store original category for category count updates
            original_category = response.category
            
            # Update the response with flag information
            response.is_flagged = True
            response.corrected_category = flag_data.corrected_category
            response.flagged_at = datetime.datetime.utcnow()
            
            await session.commit()
            
            # Get the job to get the model name
            job = await session.get(TestingJob, response.job_id)
            
            if not job:
                return {"success": True, "message": "Response flagged but unable to update category counts"}
            
            # Update category counts if the category was changed
            if original_category != flag_data.corrected_category:
                # Decrement count for original category
                if original_category:
                    result = await session.execute(
                        select(CategoryCount)
                        .where(CategoryCount.question_id == response.question_id)
                        .where(CategoryCount.category == original_category)
                        .where(CategoryCount.model_name == job.model_name)
                    )
                    
                    category_count = result.scalars().first()
                    if category_count and category_count.count > 0:
                        category_count.count -= 1
                
                # Increment or create count for corrected category
                result = await session.execute(
                    select(CategoryCount)
                    .where(CategoryCount.question_id == response.question_id)
                    .where(CategoryCount.category == flag_data.corrected_category)
                    .where(CategoryCount.model_name == job.model_name)
                )
                
                category_count = result.scalars().first()
                if category_count:
                    category_count.count += 1
                else:
                    # Create new category count
                    new_count = CategoryCount(
                        question_id=response.question_id,
                        category=flag_data.corrected_category,
                        model_name=job.model_name,
                        count=1
                    )
                    session.add(new_count)
                
                await session.commit()
            
            return {
                "success": True, 
                "message": "Response flagged and category counts updated",
                "response_id": response_id,
                "corrected_category": flag_data.corrected_category
            }
            
    except Exception as e:
        logger.error(f"Error flagging response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error flagging response: {str(e)}")
        
@router.get("/flagged_responses")
async def get_flagged_responses(model_name: str):
    """Get all flagged responses for a model"""
    try:
        async with get_db_session() as session:
            # Find the most recent job for this model
            result = await session.execute(
                select(TestingJob)
                .where(TestingJob.model_name == model_name)
                .order_by(TestingJob.id.desc())
            )
            job = result.scalars().first()
            
            if not job:
                raise HTTPException(status_code=404, detail=f"No data found for model: {model_name}")
            
            # Get all flagged responses for this job
            result = await session.execute(
                select(ModelResponse)
                .where(ModelResponse.job_id == job.id)
                .where(ModelResponse.is_flagged == True)
            )
            
            flagged_responses = result.scalars().all()
            
            # Group responses by question
            data_by_question = {}
            
            for response in flagged_responses:
                question_id = response.question_id
                question_text = next((q["text"] for q in QUESTIONS if q["id"] == question_id), "Unknown Question")
                
                if question_id not in data_by_question:
                    data_by_question[question_id] = {
                        "question_text": question_text,
                        "responses": []
                    }
                
                data_by_question[question_id]["responses"].append({
                    "id": response.id,
                    "raw_response": response.raw_response,
                    "original_category": response.category,
                    "corrected_category": response.corrected_category,
                    "flagged_at": response.flagged_at.isoformat() if response.flagged_at else None,
                    "created_at": response.created_at.isoformat() if response.created_at else None
                })
            
            # Return the flagged data
            return {
                "model_name": model_name,
                "job_id": job.id,
                "count": len(flagged_responses),
                "questions": data_by_question
            }
            
    except Exception as e:
        logger.error(f"Error retrieving flagged responses for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving flagged responses: {str(e)}")
        
@router.get("/download_flagged_responses")
async def download_flagged_responses(model_name: str):
    """Get all flagged responses for a model in a downloadable format"""
    try:
        # Reuse the get_flagged_responses function
        flagged_data = await get_flagged_responses(model_name)
        
        # Format for download
        formatted_data = {
            "model_name": flagged_data["model_name"],
            "job_id": flagged_data["job_id"],
            "total_flagged": flagged_data["count"],
            "flagged_responses": []
        }
        
        # Flatten the hierarchical structure for easier use
        for question_id, question_data in flagged_data["questions"].items():
            for response in question_data["responses"]:
                formatted_data["flagged_responses"].append({
                    "response_id": response["id"],
                    "question_id": question_id,
                    "question_text": question_data["question_text"],
                    "raw_response": response["raw_response"],
                    "original_category": response["original_category"],
                    "corrected_category": response["corrected_category"],
                    "flagged_at": response["flagged_at"],
                    "created_at": response["created_at"]
                })
        
        return formatted_data
            
    except Exception as e:
        logger.error(f"Error downloading flagged responses for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading flagged responses: {str(e)}")