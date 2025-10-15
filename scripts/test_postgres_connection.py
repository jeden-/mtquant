"""Test connection with postgres user (like pgAdmin does)."""
import psycopg2


def test_postgres_connection():
    """Test connection with postgres user."""
    try:
        print("🔑 Test połączenia z użytkownikiem 'postgres' (jak pgAdmin)...")
        
        # Try with postgres user - you need to provide the password
        password = input("Podaj hasło dla użytkownika 'postgres': ")
        
        # Connect to postgres database
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='postgres',
            user='postgres',
            password=password
        )
        
        print("✅ Połączenie z 'postgres' udane!")
        
        with conn.cursor() as cur:
            # Check PostgreSQL version
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            print(f"📊 PostgreSQL: {version}")
            
            # Check current user
            cur.execute("SELECT current_user, session_user")
            current_user, session_user = cur.fetchone()
            print(f"👤 Current user: {current_user}, Session user: {session_user}")
            
            # Check if mariusz user exists and has login rights
            cur.execute("""
                SELECT rolname, rolcanlogin, rolsuper, rolcreatedb 
                FROM pg_roles 
                WHERE rolname = 'mariusz'
            """)
            mariusz_info = cur.fetchone()
            
            if mariusz_info:
                name, can_login, is_super, can_create_db = mariusz_info
                print(f"✅ Użytkownik 'mariusz' istnieje:")
                print(f"   - Może się logować: {can_login}")
                print(f"   - Jest superuserem: {is_super}")
                print(f"   - Może tworzyć bazy: {can_create_db}")
            else:
                print("❌ Użytkownik 'mariusz' nie istnieje")
            
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
        
        # Now try to connect to mtquantum as postgres
        print(f"\n🔗 Test połączenia z bazą 'mtquantum' jako 'postgres'...")
        conn2 = psycopg2.connect(
            host='localhost',
            port=5432,
            database='mtquantum',
            user='postgres',
            password=password
        )
        
        print("✅ Połączenie z bazą 'mtquantum' jako 'postgres' udane!")
        
        with conn2.cursor() as cur:
            cur.execute("SELECT current_database(), current_user")
            db_name, user_name = cur.fetchone()
            print(f"📋 Połączono: baza='{db_name}', użytkownik='{user_name}'")
        
        conn2.close()
        
        print(f"\n🎉 SUKCES! Używamy użytkownika 'postgres' z hasłem: {password}")
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
    working_password = test_postgres_connection()
    if working_password:
        print(f"\n✅ Działające połączenie: postgres / {working_password}")
    else:
        print("\n❌ Problem z połączeniem do PostgreSQL")
