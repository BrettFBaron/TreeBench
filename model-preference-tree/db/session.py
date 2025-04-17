from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from config import DATABASE_URL, logger

# Create async engine with more conservative settings for Heroku
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    # Don't set pool_size on Heroku - it will use SQLAlchemy defaults
    # pool_size=10,
    # max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
)

# Create async session factory
async_session_factory = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

@asynccontextmanager
async def get_db_session():
    """Async context manager for database sessions with proper resource management"""
    session = async_session_factory()
    try:
        yield session
        # Don't commit here - let caller decide when to commit
    except Exception as e:
        await session.rollback()
        logger.error(f"DB session error: {str(e)}")
        raise
    finally:
        # Close session in finally block to ensure it's always closed
        await session.close()
        
# For FastAPI dependency injection - an alternative approach
async def get_session():
    """Get a database session for dependency injection"""
    async with get_db_session() as session:
        yield session

async def init_db():
    """Initialize database tables and ensure test status exists"""
    from db.models import Base, TestStatus
    
    # Create tables first
    try:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables created")
        except Exception as table_error:
            # If we get a duplicate error, just log it and continue
            # This happens when sequences exist but tables don't
            logger.warning(f"Table creation error (continuing): {str(table_error)}")
            
            # Try to fix the table if it's the unique violation error we're seeing
            if "test_status" in str(table_error) and "duplicate key value" in str(table_error):
                async with engine.begin() as conn:
                    # Drop the sequence if it exists
                    try:
                        await conn.execute(text("DROP SEQUENCE IF EXISTS test_status_id_seq"))
                        logger.info("Dropped test_status_id_seq to fix conflict")
                    except Exception as seq_error:
                        logger.warning(f"Error dropping sequence: {str(seq_error)}")

                    # Try to create just the test_status table
                    try:
                        await conn.execute(text("""
                            CREATE TABLE IF NOT EXISTS test_status (
                                id SERIAL PRIMARY KEY,
                                is_running BOOLEAN DEFAULT FALSE,
                                current_model VARCHAR(100),
                                job_id INTEGER,
                                started_at TIMESTAMP
                            )
                        """))
                        logger.info("Created test_status table manually")
                    except Exception as create_error:
                        logger.warning(f"Error creating test_status table: {str(create_error)}")
        
        # Initialize test status record if needed
        async with get_db_session() as session:
            from sqlalchemy import select, text
            
            try:
                # Check if test status record exists
                result = await session.execute(select(TestStatus))
                test_status = result.scalars().first()
                
                if not test_status:
                    # Create initial test status record
                    test_status = TestStatus(id=1, is_running=False)
                    session.add(test_status)
                    await session.commit()
                    logger.info("Test status initialized")
                else:
                    # Reset status to not running on startup
                    test_status.is_running = False
                    await session.commit()
                    logger.info("Test status reset to not running")
                    
            except Exception as status_error:
                logger.error(f"Error initializing test status: {str(status_error)}")
                # Try a more direct approach if ORM fails
                try:
                    await session.execute(text("""
                        INSERT INTO test_status (id, is_running) 
                        VALUES (1, FALSE)
                        ON CONFLICT (id) DO UPDATE SET is_running = FALSE
                    """))
                    await session.commit()
                    logger.info("Test status initialized using raw SQL")
                except Exception as raw_error:
                    logger.error(f"Error with raw SQL initialization: {str(raw_error)}")
                
        logger.info("Database initialized")
            
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        # Log but don't raise, allow the app to start anyway
        # raise