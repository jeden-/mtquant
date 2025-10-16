#!/usr/bin/env python3
"""Check broker_connections table structure."""

import asyncio
import asyncpg

async def check_table():
    conn = await asyncpg.connect('postgresql://mariusz:MARiusz@!2025@localhost:5432/mtquantum')
    
    # Get column names
    result = await conn.fetch("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'broker_connections'
        ORDER BY ordinal_position
    """)
    
    print("broker_connections table structure:")
    for row in result:
        print(f"  {row['column_name']}: {row['data_type']}")
    
    # Get sample data
    sample = await conn.fetch("SELECT * FROM broker_connections LIMIT 1")
    if sample:
        print(f"\nSample row: {dict(sample[0])}")
    else:
        print("\nNo data in table")
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(check_table())
