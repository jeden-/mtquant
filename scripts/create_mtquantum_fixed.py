"""Create mtquantum database with proper transaction handling."""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def create_mtquantum_database():
    """Create mtquantum database properly."""
    try:
        print("🔑 Łączenie z PostgreSQL jako 'postgres'...")
        
        # Get password from user
        password = input("Podaj hasło dla użytkownika 'postgres': ")
        
        # Connect to postgres database
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='postgres',
            user='postgres',
            password=password
        )
        
        print("✅ Połączono z PostgreSQL!")
        
        # Set autocommit BEFORE any operations
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            # Check if mtquantum database exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname = 'mtquantum'")
            db_exists = cur.fetchone()
            
            if db_exists:
                print("✅ Baza 'mtquantum' już istnieje")
            else:
                print("❌ Baza 'mtquantum' nie istnieje - tworzę...")
                cur.execute("CREATE DATABASE mtquantum")
                print("✅ Baza 'mtquantum' utworzona!")
        
        conn.close()
        
        # Test connection to mtquantum
        print(f"\n🔗 Test połączenia z bazą 'mtquantum'...")
        conn2 = psycopg2.connect(
            host='localhost',
            port=5432,
            database='mtquantum',
            user='postgres',
            password=password
        )
        
        print("✅ Połączenie z bazą 'mtquantum' udane!")
        
        with conn2.cursor() as cur:
            cur.execute("SELECT current_database(), current_user")
            db_name, user_name = cur.fetchone()
            print(f"📋 Połączono: baza='{db_name}', użytkownik='{user_name}'")
        
        conn2.close()
        
        print(f"\n🎉 SUKCES! Baza 'mtquantum' gotowa!")
        print(f"📝 Używamy: postgres / {password}")
        return password
        
    except psycopg2.Error as e:
        print(f"❌ Błąd PostgreSQL: {e}")
        if e.pgcode:
            print(f"   Code: {e.pgcode}")
        if e.pgerror:
            print(f"   Details: {e.pgerror}")
    except Exception as e:
        print(f"❌ Inny błąd: {e}")
    
    return None


if __name__ == "__main__":
    working_password = create_mtquantum_database()
    if working_password:
        print(f"\n✅ Gotowe! Teraz możemy inicjalizować tabele...")
    else:
        print("\n❌ Problem z utworzeniem bazy")
