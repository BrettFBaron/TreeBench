import asyncio
from sqlalchemy import update
from db.models import TestingJob
from db.session import get_db_session

async def update_all_jobs_to_completed():
    async with get_db_session() as session:
        result = await session.execute(
            update(TestingJob).where(TestingJob.status != 'completed').values(status='completed')
        )
        await session.commit()
        print(f'All jobs set to completed status. Rows affected: {result.rowcount}')

if __name__ == "__main__":
    asyncio.run(update_all_jobs_to_completed())