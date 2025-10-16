"""Create broker_connections table in PostgreSQL."""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'mtquantum',
    'user': 'postgres',
    'password': 'MARiusz@!2025'
}

def create_table():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            # Create table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS broker_connections (
                    broker_id VARCHAR(100) PRIMARY KEY,
                    broker_type VARCHAR(10) NOT NULL CHECK (broker_type IN ('mt5', 'mt4')),
                    account INTEGER NOT NULL,
                    password_encrypted VARCHAR(500) NOT NULL,
                    server VARCHAR(100) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_connected_at TIMESTAMPTZ,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE
                );
            """)
            print("✅ Tabela broker_connections utworzona!")
            
            # Create index
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_broker_connections_active 
                ON broker_connections(is_active);
            """)
            print("✅ Indeks utworzony!")
        
        conn.close()
        print("\n🎉 Gotowe! Teraz brokery będą zapisywane w bazie!")
        
    except Exception as e:
        print(f"❌ Błąd: {e}")

if __name__ == "__main__":
    create_table()

