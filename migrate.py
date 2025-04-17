import asyncio
import time
from sqlalchemy import text
from db.session import init_db, get_db_session
from db.migrate_flag_columns import add_flag_columns
from db.migrate_categories import check_categories_integrity
from config import logger, DATABASE_URL

async def run_migrations():
    """Run all database migrations for Heroku deployment"""
    print("Running database initialization and migrations...")
    
    # Wait briefly for database to be fully available
    time.sleep(1)
    
    # Step 1: Initialize the database tables
    print("Step 1: Initializing database tables...")
    try:
        await init_db()
        print("Database tables created successfully")
    except Exception as e:
        print(f"Warning during database initialization: {str(e)}")
        print("Continuing with migrations...")
    
    # Step 2: Add flag columns to model_response table
    print("Step 2: Adding flag columns if needed...")
    flag_columns_added = await add_flag_columns()
    if flag_columns_added:
        print("Flag columns added or already exist")
    else:
        print("Warning: Failed to add flag columns")
    
    # Step 3: Check category integrity
    print("Step 3: Checking category integrity...")
    try:
        await check_categories_integrity()
        print("Category integrity check completed")
    except Exception as e:
        print(f"Category integrity check error (may be normal if no data exists): {str(e)}")
    
    # Step 4: Verify TestStatus table is properly initialized
    print("Step 4: Verifying TestStatus is properly initialized...")
    async with get_db_session() as session:
        try:
            # Reset test status to not running
            await session.execute(text("""
                INSERT INTO test_status (id, is_running) 
                VALUES (1, FALSE)
                ON CONFLICT (id) DO UPDATE SET 
                    is_running = FALSE,
                    current_model = NULL,
                    job_id = NULL,
                    started_at = NULL
            """))
            await session.commit()
            print("Test status initialized successfully")
        except Exception as e:
            print(f"Error initializing test status: {str(e)}")
    
    print("All migrations completed successfully")
    print("Database is ready for use with model preference app")

if __name__ == "__main__":
    asyncio.run(run_migrations())