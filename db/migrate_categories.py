import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Import the correct database connection string
from config import DATABASE_URL, logger

async def check_categories_integrity():
    """
    Verifies the integrity of category data and ensures all the required categories
    ('refusal', 'soft_refusal', 'hedged_preference') exist in the database.
    
    This replaces the previous migration that would convert soft_refusal to hedged_preference,
    as the current code treats these as distinct categories.
    """
    print("Starting category integrity check...")
    
    # Create engine and session
    engine = create_async_engine(DATABASE_URL, echo=True)
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        # Check if all category types exist in category_count table
        required_categories = ['refusal', 'soft_refusal', 'hedged_preference']
        
        for category in required_categories:
            result = await session.execute(
                text(f"SELECT COUNT(*) FROM category_count WHERE category = '{category}'")
            )
            count = result.scalar()
            
            if count == 0:
                print(f"Category '{category}' not found in any category counts. This is expected for new databases.")
            else:
                print(f"Found {count} records for category '{category}' in category_count table.")
        
        # Check for any inconsistencies
        result = await session.execute(
            text("""
                SELECT category, COUNT(*) 
                FROM model_response 
                WHERE category NOT IN ('refusal', 'soft_refusal', 'hedged_preference') 
                GROUP BY category
            """)
        )
        
        custom_categories = result.all()
        if custom_categories:
            print("Custom preference categories found:")
            for category, count in custom_categories:
                print(f"  - {category}: {count} responses")
        else:
            print("No custom preference categories found.")
        
        print("Category integrity check completed successfully.")
        return True

# For backward compatibility, keep the old function name but update its implementation
async def migrate_soft_refusal_to_hedged_preference():
    """
    This function previously migrated 'soft_refusal' to 'hedged_preference', but
    the current codebase treats these as distinct categories. It now runs the
    category integrity check instead.
    """
    print("NOTE: The soft_refusal and hedged_preference categories are now treated as distinct.")
    print("Running category integrity check instead of migration...")
    return await check_categories_integrity()

# Run the migration
if __name__ == "__main__":
    asyncio.run(check_categories_integrity())