"""Create mtquantum database if it doesn't exist."""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def create_database():
    """Create mtquantum database."""
    try:
        print("Łączenie z PostgreSQL jako postgres...")
        
        # Connect to default 'postgres' database
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='postgres',
            user='mariusz',
            password='MARiusz@!2026'
        )
        
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        print("✅ Połączono z PostgreSQL")
        
        with conn.cursor() as cur:
            # Check if mtquantum database exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname = 'mtquantum'")
            exists = cur.fetchone()
            
            if exists:
                print("✅ Baza 'mtquantum' już istnieje")
            else:
                print("❌ Baza 'mtquantum' nie istnieje. Tworzę...")
                cur.execute("CREATE DATABASE mtquantum")
                print("✅ Baza 'mtquantum' utworzona!")
        
        conn.close()
        
        # Test connection to mtquantum
        print("\nTest połączenia z bazą 'mtquantum'...")
        conn2 = psycopg2.connect(
            host='localhost',
            port=5432,
            database='mtquantum',
            user='mariusz',
            password='MARiusz@!2026'
        )
        
        with conn2.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            print(f"✅ Połączenie z 'mtquantum' OK!")
            print(f"📊 PostgreSQL: {version}")
        
        conn2.close()
        print("\n🎉 Gotowe do inicjalizacji tabel!")
        
    except psycopg2.Error as e:
        print(f"❌ Błąd PostgreSQL: {e}")
        print(f"   Code: {e.pgcode}")
        print(f"   Details: {e.pgerror}")
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    create_database()
