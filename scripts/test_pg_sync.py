"""Synchronous PostgreSQL connection test with psycopg2."""
import psycopg2


def test_connection():
    """Test PostgreSQL connection synchronously."""
    try:
        print("Próba połączenia z PostgreSQL (psycopg2)...")
        
        # Try to connect
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='postgres',
            user='mariusz',
            password='MtQuant@!2025'
        )
        
        print("✅ Połączenie z bazą 'postgres' udane!")
        
        # Check if 'mtquant' database exists
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = 'mtquant'")
        result = cur.fetchone()
        
        if result:
            print("✅ Baza 'mtquant' już istnieje")
        else:
            print("❌ Baza 'mtquant' nie istnieje. Tworzę...")
            conn.autocommit = True
            cur.execute("CREATE DATABASE mtquant")
            print("✅ Baza 'mtquant' utworzona!")
        
        cur.close()
        conn.close()
        
        # Now try to connect to 'mtquant'
        print("\nPróba połączenia z bazą 'mtquant'...")
        conn2 = psycopg2.connect(
            host='localhost',
            port=5432,
            database='mtquant',
            user='mariusz',
            password='MtQuant@!2025'
        )
        
        print("✅ Połączenie z bazą 'mtquant' udane!")
        
        # Show PostgreSQL version
        cur2 = conn2.cursor()
        cur2.execute("SELECT version()")
        version = cur2.fetchone()[0]
        print(f"\n📊 PostgreSQL version: {version}")
        
        cur2.close()
        conn2.close()
        
        print("\n🎉 Wszystko gotowe do inicjalizacji tabel!")
        
    except psycopg2.Error as e:
        print(f"❌ Błąd PostgreSQL: {e}")
        print(f"   Code: {e.pgcode}")
        print(f"   Details: {e.pgerror}")
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_connection()


