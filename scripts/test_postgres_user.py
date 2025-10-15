"""Test connection with postgres superuser."""
import psycopg2


def test_postgres_connection():
    """Test connection with postgres user."""
    try:
        print("Test połączenia z użytkownikiem 'postgres'...")
        
        # Try with postgres user (you'll need to provide the password)
        password = input("Podaj hasło dla użytkownika 'postgres': ")
        
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='postgres',
            user='postgres',
            password=password
        )
        
        print("✅ Połączenie z 'postgres' udane!")
        
        with conn.cursor() as cur:
            # Check if mariusz user exists
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'mariusz'")
            mariusz_exists = cur.fetchone()
            
            if mariusz_exists:
                print("✅ Użytkownik 'mariusz' istnieje")
                
                # Check mariusz password
                cur.execute("SELECT rolpassword FROM pg_authid WHERE rolname = 'mariusz'")
                mariusz_pwd = cur.fetchone()
                if mariusz_pwd:
                    print("✅ Użytkownik 'mariusz' ma ustawione hasło")
                else:
                    print("❌ Użytkownik 'mariusz' nie ma hasła")
            else:
                print("❌ Użytkownik 'mariusz' nie istnieje")
            
            # Check if mtquantum database exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname = 'mtquantum'")
            db_exists = cur.fetchone()
            
            if db_exists:
                print("✅ Baza 'mtquantum' istnieje")
            else:
                print("❌ Baza 'mtquantum' nie istnieje")
        
        conn.close()
        
    except psycopg2.Error as e:
        print(f"❌ Błąd PostgreSQL: {e}")
        print(f"   Code: {e.pgcode}")
        print(f"   Details: {e.pgerror}")
    except Exception as e:
        print(f"❌ Błąd: {e}")


if __name__ == "__main__":
    test_postgres_connection()
