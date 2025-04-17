import asyncio
import random
import datetime
import httpx
from typing import Dict, List, Set, Optional, Any, Tuple
from sqlalchemy import select, update, insert, text
from sqlalchemy.ext.asyncio import AsyncSession

from config import (
    logger, QUESTIONS, ROOT_QUESTION, FOLLOWUP_QUESTION, TREE_DEPTH, 
    SAMPLES_PER_LEVEL, OPTION_COUNT, get_shuffled_options, OPENAI_API_KEY
)
from db.models import TestingJob, ModelResponse, CategoryCount, TreePathNode
from core.api_clients import get_model_response, check_category_similarity, extract_choice

class CategoryRegistry:
    """Thread-safe registry for managing categories"""
    
    def __init__(self, question_id: str, tree_level: int = 0, parent_path: Optional[str] = None):
        # Always include the incomplete category
        self._categories: Set[str] = {"incomplete"}
        self._question_id = question_id
        self._tree_level = tree_level
        self._parent_path = parent_path
        self._initialized = False
        
    def get_categories(self) -> List[str]:
        """Returns a copy of current categories"""
        return list(self._categories)
    
    def add_category(self, category: str) -> bool:
        """Adds a new category if not already present"""
        if not category:
            return False
            
        # Case insensitive check
        if not any(existing.lower() == category.lower() for existing in self._categories):
            self._categories.add(category)
            return True
        return False
    
    def normalize_category(self, category: str) -> str:
        """Finds matching category with correct capitalization"""
        if category == "incomplete":
            return "incomplete"
            
        for existing in self._categories:
            if existing.lower() == category.lower():
                return existing
        
        # No match found, return original
        return category

    async def initialize_from_db(self, session: AsyncSession) -> None:
        """Loads initial categories from database"""
        if not self._initialized:
            try:
                # Query distinct categories for this question and path context
                query = select(CategoryCount.category).distinct().where(
                    CategoryCount.question_id == self._question_id,
                    CategoryCount.tree_level == self._tree_level
                )
                
                # Add parent path filter for non-root levels
                if self._tree_level > 0 and self._parent_path:
                    query = query.where(CategoryCount.parent_path == self._parent_path)
                
                result = await session.execute(query)
                categories = [row[0] for row in result.all()]
                
                # Update categories
                for category in categories:
                    self._categories.add(category)
                
                # Always ensure the incomplete category is included
                self._categories.add("incomplete")
                
                self._initialized = True
            except Exception as e:
                logger.error(f"Error refreshing categories for {self._question_id} (level {self._tree_level}, parent: {self._parent_path}): {str(e)}")
                raise

class TreePathManager:
    """Manages tree paths during job processing"""
    
    def __init__(self, job_id: int, model_name: str):
        self.job_id = job_id
        self.model_name = model_name
        
    async def initialize_root_paths(self, session: AsyncSession) -> None:
        """Initialize root level paths"""
        # Create a TreePathNode for each possible root category (including incomplete)
        # We'll start with all paths as active, then mark inactive ones after level 0 processing
        # Use actual option names from configuration instead of generic placeholders
        possible_root_categories = ["incomplete"] + ROOT_QUESTION["options"]
        
        for category in possible_root_categories:
            node = TreePathNode(
                job_id=self.job_id,
                question_id=ROOT_QUESTION["id"],
                tree_level=0,
                parent_path=None,
                category=category,
                is_active=True,
                sample_count=0
            )
            session.add(node)
        
        await session.commit()
        logger.info(f"Initialized {len(possible_root_categories)} root paths for job {self.job_id}")
    
    async def mark_active_paths(self, session: AsyncSession, tree_level: int) -> Dict[str, List[str]]:
        """
        Mark paths as active/inactive based on responses at current level
        Returns a dict mapping parent paths to active categories
        """
        # Get all responses for this job at the current level
        result = await session.execute(
            select(ModelResponse)
            .where(
                ModelResponse.job_id == self.job_id,
                ModelResponse.tree_level == tree_level
            )
        )
        responses = result.scalars().all()
        
        # Group responses by parent path (for root level, parent_path is None)
        grouped_categories: Dict[str, Dict[str, int]] = {}
        
        for response in responses:
            parent_key = response.parent_path or "root"
            if parent_key not in grouped_categories:
                grouped_categories[parent_key] = {}
                
            category = response.category
            if category not in grouped_categories[parent_key]:
                grouped_categories[parent_key][category] = 0
                
            grouped_categories[parent_key][category] += 1
        
        # Now determine active paths for the next level
        active_paths: Dict[str, List[str]] = {}
        
        for parent_key, categories in grouped_categories.items():
            parent_path = None if parent_key == "root" else parent_key
            
            # Get actual categories that appeared in responses (excluding incomplete)
            valid_categories = [
                cat for cat, count in categories.items()
                if cat != "incomplete" and count > 0
            ]
            
            if valid_categories:
                parent_path_str = parent_path or ""
                active_paths[parent_path_str] = valid_categories
                
                # Update TreePathNode records
                for category in valid_categories:
                    # First, check if this path node exists
                    path_query = select(TreePathNode).where(
                        TreePathNode.job_id == self.job_id,
                        TreePathNode.tree_level == tree_level,
                        TreePathNode.category == category
                    )
                    
                    if parent_path:
                        path_query = path_query.where(TreePathNode.parent_path == parent_path)
                    else:
                        path_query = path_query.where(TreePathNode.parent_path.is_(None))
                        
                    result = await session.execute(path_query)
                    path_node = result.scalars().first()
                    
                    # Create or update path node
                    if path_node:
                        # Update existing node
                        path_node.is_active = True
                        path_node.sample_count = categories.get(category, 0)
                    else:
                        # Create new node
                        new_node = TreePathNode(
                            job_id=self.job_id,
                            question_id=ROOT_QUESTION["id"] if tree_level == 0 else FOLLOWUP_QUESTION["id"],
                            tree_level=tree_level,
                            parent_path=parent_path,
                            category=category,
                            is_active=True,
                            sample_count=categories.get(category, 0)
                        )
                        session.add(new_node)
        
        # Mark inactive paths
        all_path_query = select(TreePathNode).where(
            TreePathNode.job_id == self.job_id,
            TreePathNode.tree_level == tree_level
        )
        result = await session.execute(all_path_query)
        all_paths = result.scalars().all()
        
        for path in all_paths:
            parent_key = "root" if path.parent_path is None else path.parent_path
            
            # Mark as inactive if:
            # 1. Parent path not in grouped_categories (no responses for this parent)
            # 2. Category not in grouped_categories[parent_key] (no responses with this category)
            # 3. Category is "incomplete" (we don't follow up on incomplete responses)
            if (
                parent_key not in grouped_categories or 
                path.category not in grouped_categories[parent_key] or
                path.category == "incomplete"
            ):
                path.is_active = False
        
        await session.commit()
        logger.info(f"Marked active paths for level {tree_level}: {active_paths}")
        return active_paths
    
    async def get_next_level_paths(self, session: AsyncSession, current_level: int) -> List[Tuple[str, str]]:
        """Get active paths for the next tree level"""
        # Get active paths from current level
        result = await session.execute(
            select(TreePathNode)
            .where(
                TreePathNode.job_id == self.job_id,
                TreePathNode.tree_level == current_level,
                TreePathNode.is_active == True,
                TreePathNode.category != "incomplete"  # Skip incomplete responses
            )
        )
        active_nodes = result.scalars().all()
        
        # Convert to parent paths for next level
        next_level_paths = []
        
        for node in active_nodes:
            # Construct path for next level
            if current_level == 0:
                # Root level - path is just the category
                path = node.category
            else:
                # Non-root level - path is parent_path→category
                path = f"{node.parent_path}→{node.category}"
                
            # Create (parent_path, category) tuple
            next_level_paths.append((path, node.category))
        
        return next_level_paths
    
    async def allocate_samples(self, session: AsyncSession, level: int, total_samples: int) -> Dict[Tuple[str, str], int]:
        """
        Allocate samples across active paths for a given level
        Returns a dict mapping (parent_path, last_category) to number of samples
        
        Sampling strategy:
        - Level 0: Sample exactly 32 times total
        - Level 1: Sample exactly 32 times for each active node from level 0
        """
        # Get active paths for this level
        if level == 0:
            # For root level, sample exactly 32 times total
            logger.info(f"Using exactly 32 samples for level 0")
            return {(None, None): 32}
        
        # For non-root levels, get active paths from previous level
        active_paths = await self.get_next_level_paths(session, level - 1)
        
        if not active_paths:
            logger.warning(f"No active paths found for level {level}")
            return {}
            
        # For level 1, use exactly 32 samples per node
        logger.info(f"Using exactly 32 samples per path for level 1 across {len(active_paths)} paths")
        
        # Allocate samples
        allocation = {}
        for path, category in active_paths:
            allocation[(path, category)] = 32
            
        return allocation

async def process_job(job_id: int, model_data: Dict[str, Any]) -> bool:
    """
    Process a job with tree-based structure
    """
    from db.session import get_db_session
    
    try:
        logger.info(f"Starting tree-based job {job_id}")
        
        # Extract API details
        model_name = model_data['model_name']
        api_url = model_data['api_url']
        api_key = model_data['api_key']
        api_type = model_data['api_type']
        model_id = model_data['model_id']
        
        # Update job status and reset tree level
        async with get_db_session() as session:
            job = await session.get(TestingJob, job_id)
            if job:
                job.status = "running"
                job.current_tree_level = 0
                await session.commit()
        
        # Create OpenAI client for all classification tasks
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured in .env file. This is required for all classifications.")
            
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
        
        # Initialize TreePathManager
        path_manager = TreePathManager(job_id, model_name)
        
        # Initialize root paths
        async with get_db_session() as session:
            await path_manager.initialize_root_paths(session)
        
        # Process each level of the tree sequentially
        for level in range(TREE_DEPTH):
            logger.info(f"Processing tree level {level} for job {job_id}")
            
            # Update job's current tree level
            async with get_db_session() as session:
                job = await session.get(TestingJob, job_id)
                if job:
                    job.current_tree_level = level
                    await session.commit()
            
            # Process this tree level
            level_success = await process_tree_level(
                job_id, model_name, level, path_manager,
                api_url, api_key, api_type, model_id, openai_client
            )
            
            if not level_success:
                logger.error(f"Failed to process tree level {level} for job {job_id}")
                
                # Mark job as failed
                async with get_db_session() as session:
                    job = await session.get(TestingJob, job_id)
                    if job:
                        job.status = "failed"
                        job.completed_at = datetime.datetime.utcnow()
                        await session.commit()
                        
                # Clean up client and return
                await openai_client.aclose()
                return False
            
            # After each level, mark active paths for the next level
            async with get_db_session() as session:
                await path_manager.mark_active_paths(session, level)
        
        # All levels processed successfully
        async with get_db_session() as session:
            job = await session.get(TestingJob, job_id)
            if job:
                job.status = "completed"
                job.completed_at = datetime.datetime.utcnow()
                await session.commit()
                
                # Update test status to indicate no test is running
                from sqlalchemy import select
                from db.models import TestStatus
                result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
                test_status = result.scalars().first()
                
                if test_status:
                    test_status.is_running = False
                    await session.commit()
        
        # Clean up client
        await openai_client.aclose()
        
        logger.info(f"Job {job_id} completed successfully")
        return True
        
    except Exception as e:
        logger.exception(f"Error processing job {job_id}: {str(e)}")
        
        # Mark job as failed in database
        async with get_db_session() as session:
            job = await session.get(TestingJob, job_id)
            if job:
                job.status = "failed"
                job.completed_at = datetime.datetime.utcnow()
                await session.commit()
                
            # Update test status to indicate no test is running
            from sqlalchemy import select
            from db.models import TestStatus
            result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
            test_status = result.scalars().first()
            
            if test_status:
                test_status.is_running = False
                await session.commit()
        
        return False

async def process_tree_level(
    job_id: int,
    model_name: str,
    tree_level: int,
    path_manager: TreePathManager,
    api_url: str,
    api_key: str,
    api_type: str,
    model_id: str,
    openai_client: httpx.AsyncClient
) -> bool:
    """Process all samples for a single tree level"""
    from db.session import get_db_session
    
    try:
        logger.info(f"Processing tree level {tree_level} for job {job_id}")
        
        # Get sample allocation for this level
        async with get_db_session() as session:
            sample_allocation = await path_manager.allocate_samples(
                session, tree_level, SAMPLES_PER_LEVEL
            )
        
        if not sample_allocation:
            logger.warning(f"No samples to process for tree level {tree_level}")
            return True  # Consider this a success since there's nothing to do
        
        # Create tasks for each path in the allocation
        tasks = []
        
        for (parent_path, parent_category), num_samples in sample_allocation.items():
            # Skip if no samples allocated
            if num_samples <= 0:
                continue
                
            # Use the appropriate question template based on level
            if tree_level == 0:
                question_template = ROOT_QUESTION
            else:
                question_template = FOLLOWUP_QUESTION
            
            # Create task for this path
            task = asyncio.create_task(
                process_tree_path(
                    job_id, model_name, tree_level,
                    parent_path, parent_category, num_samples,
                    question_template, api_url, api_key, api_type, model_id,
                    openai_client
                )
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check if any tasks failed
        failed_paths = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                path_info = list(sample_allocation.keys())[i]
                logger.error(f"Error processing path {path_info}: {str(result)}")
                failed_paths.append(path_info)
            elif not result:
                path_info = list(sample_allocation.keys())[i]
                failed_paths.append(path_info)
        
        # Consider level processing successful if at least one path succeeded
        level_success = len(failed_paths) < len(sample_allocation)
        
        if level_success:
            logger.info(f"Tree level {tree_level} processed successfully for job {job_id}")
        else:
            logger.error(f"All paths failed for tree level {tree_level} for job {job_id}")
        
        return level_success
        
    except Exception as e:
        logger.exception(f"Error processing tree level {tree_level} for job {job_id}: {str(e)}")
        return False

async def process_tree_path(
    job_id: int,
    model_name: str,
    tree_level: int,
    parent_path: Optional[str],
    parent_category: Optional[str],
    num_samples: int,
    question_template: Dict[str, Any],
    api_url: str,
    api_key: str,
    api_type: str,
    model_id: str,
    openai_client: httpx.AsyncClient
) -> bool:
    """Process samples for a specific path in the tree"""
    from db.session import get_db_session
    
    # For OpenRouter models, use specialized sampling that ensures exactly 32 valid samples
    if api_type == "openrouter":
        return await process_openrouter_tree_path(
            job_id, model_name, tree_level, parent_path, parent_category, 
            num_samples, question_template, api_url, api_key, api_type, 
            model_id, openai_client
        )
    
    # For all other API types, continue with the original logic
    # Determine question ID
    question_id = question_template["id"]
    
    logger.info(f"Processing {num_samples} samples for path level={tree_level}, parent={parent_path} in job {job_id}")
    
    try:
        # Initialize category registry for this path
        category_registry = CategoryRegistry(question_id, tree_level, parent_path)
        
        # Initialize from database
        async with get_db_session() as session:
            await category_registry.initialize_from_db(session)
        
        success_count = 0
        failure_count = 0
        
        # Process each sample for this path
        for sample_idx in range(num_samples):
            try:
                # Check if job has been canceled
                async with get_db_session() as session:
                    # Check job status
                    job = await session.get(TestingJob, job_id)
                    if job and job.status == "canceled":
                        logger.info(f"Job {job_id} has been canceled, stopping processing for path level={tree_level}, parent={parent_path}")
                        return False
                    
                    # Also check test status
                    from sqlalchemy import select
                    from db.models import TestStatus
                    result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
                    test_status = result.scalars().first()
                    if test_status and not test_status.is_running:
                        logger.info(f"Test status shows not running, stopping processing for path level={tree_level}, parent={parent_path}")
                        return False
                
                logger.info(f"Processing sample {sample_idx+1}/{num_samples} for path level={tree_level}, parent={parent_path}")
                
                # Construct question text with shuffled options
                question_text = question_template["text"]
                
                # For level 1 questions, insert the parent choice into the template
                if tree_level > 0 and parent_category and "{{previous_choice}}" in question_text:
                    question_text = question_text.replace("{{previous_choice}}", parent_category)
                
                # Get a shuffled and labeled version of the options
                shuffled_options = get_shuffled_options(question_template["options"])
                
                # Replace options placeholder in question text
                if "{{options}}" in question_text:
                    question_text = question_text.replace("{{options}}", shuffled_options)
                
                # Store the original question options (before shuffling) for tracking
                original_options = question_template["options"]
                
                # Store the original-to-shuffled mapping for analysis
                option_mapping = {}
                shuffled_lines = shuffled_options.split('\n')
                for i, line in enumerate(shuffled_lines):
                    # Extract the option from lines like "a) France"
                    option_text = line[3:]  # Skip "a) "
                    position = chr(97 + i)  # a, b, c, etc.
                    original_index = original_options.index(option_text)
                    option_mapping[f"option_{original_index}"] = position
                
                # Log the option mapping and shuffled options for debugging
                logger.info(f"Option mapping for sample {sample_idx}: {option_mapping}")
                logger.info(f"Shuffled options for sample {sample_idx}: \n{shuffled_options}")
                
                # Get model response
                raw_response = await get_model_response(
                    api_url, api_key, api_type, model_id, question_text
                )
                
                # Classification system - assume all responses are complete
                # Extract the specific choice that was made
                choice = await extract_choice(raw_response, question_text, openai_client)
                
                # Check similarity with existing categories
                current_categories = category_registry.get_categories()
                preference_categories = [cat for cat in current_categories 
                                       if cat != "incomplete"]
                
                if len(preference_categories) == 0:
                    # No existing categories, use the choice directly
                    category = choice
                else:
                    # Check similarity with existing categories
                    category = await check_category_similarity(
                        choice, preference_categories, openai_client
                    )
                
                # Get normalized category name
                normalized_category = category_registry.normalize_category(category)
                
                # If this is a new category, add it to registry
                if normalized_category == category and category != "incomplete":
                    category_registry.add_category(category)
                
                # Store in database
                async with get_db_session() as session:
                    try:
                        # Create full path context for this response
                        path_context = None
                        if tree_level == 0:
                            path_context = normalized_category
                        else:
                            path_context = f"{parent_path}→{normalized_category}"
                        
                        # Convert option mapping to JSON string
                        import json
                        option_mapping_json = json.dumps(option_mapping)
                        
                        # Create response object
                        response = ModelResponse(
                            job_id=job_id,
                            question_id=question_id,
                            raw_response=raw_response,
                            category=normalized_category,
                            tree_level=tree_level,
                            parent_path=parent_path,
                            path_context=path_context,
                            option_mapping=option_mapping_json,
                            shuffled_options=shuffled_options
                        )
                        session.add(response)
                        
                        # Update category count - use a more robust approach with direct SQL to handle race conditions
                        try:
                            # First try to update existing count
                            update_stmt = text("""
                            UPDATE category_count 
                            SET count = count + 1
                            WHERE question_id = :question_id
                            AND category = :category
                            AND model_name = :model_name
                            AND tree_level = :tree_level
                            AND (parent_path = :parent_path OR 
                                (parent_path IS NULL AND :parent_path IS NULL))
                            """)
                            
                            result = await session.execute(update_stmt, {
                                "question_id": question_id,
                                "category": normalized_category,
                                "model_name": model_name,
                                "tree_level": tree_level,
                                "parent_path": parent_path
                            })
                            
                            # If no rows were updated, we need to insert
                            if result.rowcount == 0:
                                # Create new category count record
                                category_count = CategoryCount(
                                    question_id=question_id,
                                    category=normalized_category,
                                    model_name=model_name,
                                    count=1,
                                    tree_level=tree_level,
                                    parent_path=parent_path
                                )
                                session.add(category_count)
                        except Exception as category_error:
                            logger.error(f"Error updating category count: {category_error}")
                            # Continue processing even if category count update fails
                        
                        # Commit transaction
                        await session.commit()
                        
                        # Success
                        success_count += 1
                        
                    except Exception as db_error:
                        await session.rollback()
                        logger.error(f"Database error for sample {sample_idx} on path level={tree_level}, parent={parent_path}: {str(db_error)}")
                        failure_count += 1
                
                # Add a small delay between API calls to avoid rate limiting
                if api_type == "anthropic":
                    await asyncio.sleep(random.uniform(0.3, 0.6))
                else:
                    await asyncio.sleep(random.uniform(0.6, 1.2))
                
            except Exception as e:
                logger.error(f"Error processing sample {sample_idx} for path level={tree_level}, parent={parent_path}: {str(e)}")
                failure_count += 1
        
        # Check if path processing was successful (80% success rate is acceptable)
        if num_samples > 0:
            path_success_rate = success_count / num_samples
            
            if path_success_rate >= 0.8:
                logger.info(f"Path level={tree_level}, parent={parent_path} completed successfully with {success_count}/{num_samples} samples")
                succeeded = True
            else:
                logger.error(f"Path level={tree_level}, parent={parent_path} failed with only {success_count}/{num_samples} samples")
                succeeded = False
        else:
            # No samples to process
            succeeded = True
        
        return succeeded
        
    except Exception as e:
        logger.exception(f"Error processing path level={tree_level}, parent={parent_path}: {str(e)}")
        return False

async def process_openrouter_tree_path(
    job_id: int,
    model_name: str,
    tree_level: int,
    parent_path: Optional[str],
    parent_category: Optional[str],
    num_samples: int,
    question_template: Dict[str, Any],
    api_url: str,
    api_key: str,
    api_type: str,
    model_id: str,
    openai_client: httpx.AsyncClient
) -> bool:
    """Process samples for OpenRouter models, ensuring exactly num_samples valid responses"""
    from db.session import get_db_session
    
    # Determine question ID
    question_id = question_template["id"]
    
    logger.info(f"Processing OpenRouter samples: targeting {num_samples} valid responses for path level={tree_level}, parent={parent_path} in job {job_id}")
    
    try:
        # Initialize category registry for this path
        category_registry = CategoryRegistry(question_id, tree_level, parent_path)
        
        # Initialize from database
        async with get_db_session() as session:
            await category_registry.initialize_from_db(session)
        
        # Track progress
        success_count = 0
        failure_count = 0
        empty_response_count = 0
        valid_sample_count = 0
        sample_idx = 0
        
        # Continue until we have enough valid samples
        while valid_sample_count < num_samples:
            try:
                # Check if job has been canceled
                async with get_db_session() as session:
                    # Check job status
                    job = await session.get(TestingJob, job_id)
                    if job and job.status == "canceled":
                        logger.info(f"Job {job_id} has been canceled, stopping processing for path level={tree_level}, parent={parent_path}")
                        return False
                    
                    # Also check test status
                    from sqlalchemy import select
                    from db.models import TestStatus
                    result = await session.execute(select(TestStatus).where(TestStatus.id == 1))
                    test_status = result.scalars().first()
                    if test_status and not test_status.is_running:
                        logger.info(f"Test status shows not running, stopping processing for path level={tree_level}, parent={parent_path}")
                        return False
                
                logger.info(f"Processing sample {sample_idx+1} (valid: {valid_sample_count}/{num_samples}) for path level={tree_level}, parent={parent_path}")
                
                # Construct question text with shuffled options
                question_text = question_template["text"]
                
                # For level 1 questions, insert the parent choice into the template
                if tree_level > 0 and parent_category and "{{previous_choice}}" in question_text:
                    question_text = question_text.replace("{{previous_choice}}", parent_category)
                
                # Get a shuffled and labeled version of the options
                shuffled_options = get_shuffled_options(question_template["options"])
                
                # Replace options placeholder in question text
                if "{{options}}" in question_text:
                    question_text = question_text.replace("{{options}}", shuffled_options)
                
                # Store the original question options (before shuffling) for tracking
                original_options = question_template["options"]
                
                # Store the original-to-shuffled mapping for analysis
                option_mapping = {}
                shuffled_lines = shuffled_options.split('\n')
                for i, line in enumerate(shuffled_lines):
                    # Extract the option from lines like "a) France"
                    option_text = line[3:]  # Skip "a) "
                    position = chr(97 + i)  # a, b, c, etc.
                    original_index = original_options.index(option_text)
                    option_mapping[f"option_{original_index}"] = position
                
                # Log the option mapping and shuffled options for debugging
                logger.info(f"Option mapping for sample {sample_idx}: {option_mapping}")
                logger.info(f"Shuffled options for sample {sample_idx}: \n{shuffled_options}")
                
                # Get model response
                raw_response = await get_model_response(
                    api_url, api_key, api_type, model_id, question_text
                )
                
                # Check if response is empty
                if not raw_response or raw_response.strip() == "":
                    logger.warning(f"OpenRouter empty response ({sample_idx+1}): continuing until {num_samples} valid samples")
                    empty_response_count += 1
                    sample_idx += 1
                    continue  # Skip empty responses
                
                # Process this valid response
                choice = await extract_choice(raw_response, question_text, openai_client)
                
                # Check similarity with existing categories
                current_categories = category_registry.get_categories()
                preference_categories = [cat for cat in current_categories 
                                      if cat != "incomplete"]
                
                if len(preference_categories) == 0:
                    # No existing categories, use the choice directly
                    category = choice
                else:
                    # Check similarity with existing categories
                    category = await check_category_similarity(
                        choice, preference_categories, openai_client
                    )
                
                # Get normalized category name
                normalized_category = category_registry.normalize_category(category)
                
                # If this is a new category, add it to registry
                if normalized_category == category and category != "incomplete":
                    category_registry.add_category(category)
                
                # Store in database
                async with get_db_session() as session:
                    try:
                        # Create full path context for this response
                        path_context = None
                        if tree_level == 0:
                            path_context = normalized_category
                        else:
                            path_context = f"{parent_path}→{normalized_category}"
                        
                        # Convert option mapping to JSON string
                        import json
                        option_mapping_json = json.dumps(option_mapping)
                        
                        # Create response object
                        response = ModelResponse(
                            job_id=job_id,
                            question_id=question_id,
                            raw_response=raw_response,
                            category=normalized_category,
                            tree_level=tree_level,
                            parent_path=parent_path,
                            path_context=path_context,
                            option_mapping=option_mapping_json,
                            shuffled_options=shuffled_options
                        )
                        session.add(response)
                        
                        # Update category count - use a more robust approach with direct SQL to handle race conditions
                        try:
                            # First try to update existing count
                            update_stmt = text("""
                            UPDATE category_count 
                            SET count = count + 1
                            WHERE question_id = :question_id
                            AND category = :category
                            AND model_name = :model_name
                            AND tree_level = :tree_level
                            AND (parent_path = :parent_path OR 
                                (parent_path IS NULL AND :parent_path IS NULL))
                            """)
                            
                            result = await session.execute(update_stmt, {
                                "question_id": question_id,
                                "category": normalized_category,
                                "model_name": model_name,
                                "tree_level": tree_level,
                                "parent_path": parent_path
                            })
                            
                            # If no rows were updated, we need to insert
                            if result.rowcount == 0:
                                # Create new category count record
                                category_count = CategoryCount(
                                    question_id=question_id,
                                    category=normalized_category,
                                    model_name=model_name,
                                    count=1,
                                    tree_level=tree_level,
                                    parent_path=parent_path
                                )
                                session.add(category_count)
                        except Exception as category_error:
                            logger.error(f"Error updating category count: {category_error}")
                            # Continue processing even if category count update fails
                        
                        # Commit transaction
                        await session.commit()
                        
                        # Success
                        success_count += 1
                        valid_sample_count += 1  # Increment valid sample count
                        
                    except Exception as db_error:
                        await session.rollback()
                        logger.error(f"Database error for sample {sample_idx} on path level={tree_level}, parent={parent_path}: {str(db_error)}")
                        failure_count += 1
                
                # Add a small delay between API calls to avoid rate limiting
                await asyncio.sleep(random.uniform(0.6, 1.2))
                
                # Increment sample counter
                sample_idx += 1
                
            except Exception as e:
                logger.error(f"Error processing sample {sample_idx} for path level={tree_level}, parent={parent_path}: {str(e)}")
                failure_count += 1
                sample_idx += 1  # Still increment the counter
        
        # Log final stats
        if empty_response_count > 0:
            logger.info(f"Path had {empty_response_count} empty responses which were skipped and replaced")
        
        logger.info(f"Path level={tree_level}, parent={parent_path} completed with {success_count}/{num_samples} valid samples")
        
        # For OpenRouter, we always consider the path successful if we got all valid samples
        return True
        
    except Exception as e:
        logger.exception(f"Error processing path level={tree_level}, parent={parent_path}: {str(e)}")
        return False

async def clear_existing_model_data(model_name: str) -> bool:
    """Clear existing data for a model from database"""
    from db.session import get_db_session
    
    logger.info(f"Clearing existing data for model: {model_name}")
    try:
        async with get_db_session() as session:
            # Find all jobs for this model
            result = await session.execute(
                select(TestingJob).where(TestingJob.model_name == model_name)
            )
            jobs = result.scalars().all()
            
            # Delete category counts
            await session.execute(
                CategoryCount.__table__.delete().where(CategoryCount.model_name == model_name)
            )
            
            # Delete jobs (which will cascade delete responses and tree paths)
            for job in jobs:
                await session.delete(job)
            
            # Commit the changes
            await session.commit()
            
            logger.info(f"Cleared existing data for model: {model_name}")
            return True
    except Exception as e:
        logger.error(f"Error clearing data for model {model_name}: {str(e)}")
        return False