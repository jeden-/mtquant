"""Test connection with mariusz user - different passwords."""
import psycopg2


def test_mariusz_connection():
    """Test connection with mariusz user."""
    
    passwords = [
        'MARiusz@!2025',  # Original
        'MARiusz@!2026',  # With 6
    ]
    
    for password in passwords:
        try:
            print(f"\n🔑 Testowanie hasła: {password}")
            
            # Try to connect to postgres database first
            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                database='postgres',
                user='mariusz',
                password=password
            )
            
            print(f"✅ Połączenie z bazą 'postgres' udane!")
            
            with conn.cursor() as cur:
                # Check PostgreSQL version
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
                print(f"📊 PostgreSQL: {version}")
                
                # Check if mtquantum database exists
                cur.execute("SELECT 1 FROM pg_database WHERE datname = 'mtquantum'")
                db_exists = cur.fetchone()
                
                if db_exists:
                    print("✅ Baza 'mtquantum' istnieje")
                else:
                    print("❌ Baza 'mtquantum' nie istnieje - tworzę...")
                    conn.autocommit = True
                    cur.execute("CREATE DATABASE mtquantum")
                    print("✅ Baza 'mtquantum' utworzona!")
            
            conn.close()
            
            # Now try to connect to mtquantum
            print(f"\n🔗 Test połączenia z bazą 'mtquantum'...")
            conn2 = psycopg2.connect(
                host='localhost',
                port=5432,
                database='mtquantum',
                user='mariusz',
                password=password
            )
            
            print(f"✅ Połączenie z bazą 'mtquantum' udane!")
            
            with conn2.cursor() as cur:
                cur.execute("SELECT current_database(), current_user")
                db_name, user_name = cur.fetchone()
                print(f"📋 Połączono: baza='{db_name}', użytkownik='{user_name}'")
            
            conn2.close()
            
            print(f"\n🎉 SUKCES! Hasło: {password}")
            return password
            
        except psycopg2.Error as e:
            print(f"❌ Błąd z hasłem {password}: {e}")
            if e.pgcode:
                print(f"   Code: {e.pgcode}")
            if e.pgerror:
                print(f"   Details: {e.pgerror}")
        except Exception as e:
            print(f"❌ Inny błąd z hasłem {password}: {e}")
    
    print("\n❌ Żadne hasło nie zadziałało!")
    return None


if __name__ == "__main__":
    working_password = test_mariusz_connection()
    if working_password:
        print(f"\n✅ Działające hasło: {working_password}")
    else:
        print("\n❌ Problem z połączeniem do PostgreSQL")
