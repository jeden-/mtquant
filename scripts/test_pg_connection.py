"""Quick PostgreSQL connection test."""
import asyncio
import asyncpg


async def test_connection():
    """Test PostgreSQL connection."""
    try:
        print("Próba połączenia z PostgreSQL...")
        
        # Try to connect to default 'postgres' database first
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            database='postgres',
            user='mariusz',
            password='MtQuant@!2025',
            timeout=5
        )
        
        print("✅ Połączenie z bazą 'postgres' udane!")
        
        # Check if 'mtquant' database exists
        result = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = 'mtquant'"
        )
        
        if result:
            print("✅ Baza 'mtquant' już istnieje")
        else:
            print("❌ Baza 'mtquant' nie istnieje. Tworzę...")
            await conn.execute("CREATE DATABASE mtquant")
            print("✅ Baza 'mtquant' utworzona!")
        
        await conn.close()
        
        # Now try to connect to 'mtquant'
        print("\nPróba połączenia z bazą 'mtquant'...")
        conn2 = await asyncpg.connect(
            host='localhost',
            port=5432,
            database='mtquant',
            user='mariusz',
            password='MtQuant@!2025',
            timeout=5
        )
        
        print("✅ Połączenie z bazą 'mtquant' udane!")
        await conn2.close()
        
        print("\n🎉 Wszystko gotowe do inicjalizacji tabel!")
        
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_connection())

