"""
PostgreSQL Database Initialization Script

Creates all required tables for MTQuant production system.
Run this once after PostgreSQL installation.

Usage:
    python scripts/init_postgresql.py
"""

import asyncio
import asyncpg
from loguru import logger

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'mtquant',
    'user': 'mariusz',
    'password': 'MtQuant@!2025'
}


async def create_tables(conn: asyncpg.Connection):
    """Create all required tables."""
    
    logger.info("Creating tables...")
    
    # 1. Orders table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id BIGSERIAL PRIMARY KEY,
            agent_id VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
            order_type VARCHAR(20) NOT NULL,
            quantity DECIMAL(18, 8) NOT NULL,
            price DECIMAL(18, 8),
            status VARCHAR(20) NOT NULL,
            broker_order_id VARCHAR(100),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_by VARCHAR(100) NOT NULL DEFAULT 'system'
        );
    """)
    logger.success("✓ Table 'orders' created")
    
    # 2. Trades table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id BIGSERIAL PRIMARY KEY,
            order_id BIGINT REFERENCES orders(order_id),
            agent_id VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
            quantity DECIMAL(18, 8) NOT NULL,
            price DECIMAL(18, 8) NOT NULL,
            commission DECIMAL(18, 8) NOT NULL DEFAULT 0,
            realized_pnl DECIMAL(18, 8),
            executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            broker_trade_id VARCHAR(100)
        );
    """)
    logger.success("✓ Table 'trades' created")
    
    # 3. Positions table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            position_id BIGSERIAL PRIMARY KEY,
            agent_id VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            side VARCHAR(4) NOT NULL CHECK (side IN ('long', 'short')),
            quantity DECIMAL(18, 8) NOT NULL,
            entry_price DECIMAL(18, 8) NOT NULL,
            current_price DECIMAL(18, 8) NOT NULL,
            unrealized_pnl DECIMAL(18, 8) NOT NULL,
            opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            closed_at TIMESTAMPTZ,
            status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
            UNIQUE(agent_id, symbol, status)
        );
    """)
    logger.success("✓ Table 'positions' created")
    
    # 4. Agent configuration table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_config (
            agent_id VARCHAR(50) PRIMARY KEY,
            config JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    logger.success("✓ Table 'agent_config' created")
    
    # 5. Audit log table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            log_id BIGSERIAL PRIMARY KEY,
            event_type VARCHAR(50) NOT NULL,
            user_id VARCHAR(100) NOT NULL DEFAULT 'system',
            agent_id VARCHAR(50),
            symbol VARCHAR(20),
            action TEXT,
            details JSONB,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    logger.success("✓ Table 'audit_log' created")
    
    # 6. Performance metrics table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            metric_id BIGSERIAL PRIMARY KEY,
            agent_id VARCHAR(50) NOT NULL,
            date DATE NOT NULL,
            total_trades INTEGER NOT NULL DEFAULT 0,
            winning_trades INTEGER NOT NULL DEFAULT 0,
            losing_trades INTEGER NOT NULL DEFAULT 0,
            total_pnl DECIMAL(18, 8) NOT NULL DEFAULT 0,
            win_rate DECIMAL(5, 4),
            sharpe_ratio DECIMAL(10, 6),
            max_drawdown DECIMAL(10, 6),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(agent_id, date)
        );
    """)
    logger.success("✓ Table 'performance_metrics' created")


async def create_indexes(conn: asyncpg.Connection):
    """Create indexes for performance optimization."""
    
    logger.info("Creating indexes...")
    
    # Orders indexes
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_agent_symbol ON orders(agent_id, symbol);")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at DESC);")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);")
    logger.success("✓ Orders indexes created")
    
    # Trades indexes
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_agent_symbol ON trades(agent_id, symbol);")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_executed_at ON trades(executed_at DESC);")
    logger.success("✓ Trades indexes created")
    
    # Positions indexes
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_agent_symbol ON positions(agent_id, symbol);")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);")
    logger.success("✓ Positions indexes created")
    
    # Agent config JSONB index
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_config_jsonb ON agent_config USING GIN(config);")
    logger.success("✓ Agent config indexes created")
    
    # Audit log indexes
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type);")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp DESC);")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_agent ON audit_log(agent_id);")
    logger.success("✓ Audit log indexes created")
    
    # Performance metrics indexes
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_agent_date ON performance_metrics(agent_id, date DESC);")
    logger.success("✓ Performance metrics indexes created")


async def create_triggers(conn: asyncpg.Connection):
    """Create triggers for automatic timestamp updates."""
    
    logger.info("Creating triggers...")
    
    # Function for updating 'updated_at' column
    await conn.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    # Triggers for orders, positions, agent_config
    await conn.execute("""
        CREATE TRIGGER update_orders_updated_at 
        BEFORE UPDATE ON orders 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    
    await conn.execute("""
        CREATE TRIGGER update_positions_updated_at 
        BEFORE UPDATE ON positions 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    
    await conn.execute("""
        CREATE TRIGGER update_agent_config_updated_at 
        BEFORE UPDATE ON agent_config 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    
    logger.success("✓ Triggers created")


async def insert_sample_data(conn: asyncpg.Connection):
    """Insert sample configuration for testing."""
    
    logger.info("Inserting sample data...")
    
    # Sample agent configuration
    await conn.execute("""
        INSERT INTO agent_config (agent_id, config)
        VALUES ($1, $2)
        ON CONFLICT (agent_id) DO NOTHING;
    """, 'XAUUSD_agent', {
        'symbol': 'XAUUSD',
        'max_position_size': 1.0,
        'risk_per_trade': 0.02,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'trading_style': 'day_trader'
    })
    
    logger.success("✓ Sample data inserted")


async def verify_setup(conn: asyncpg.Connection):
    """Verify database setup."""
    
    logger.info("Verifying setup...")
    
    # Check all tables exist
    tables = await conn.fetch("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    
    logger.info(f"Found {len(tables)} tables:")
    for table in tables:
        logger.info(f"  - {table['table_name']}")
    
    # Check indexes
    indexes = await conn.fetch("""
        SELECT indexname 
        FROM pg_indexes 
        WHERE schemaname = 'public'
        ORDER BY indexname;
    """)
    
    logger.info(f"Found {len(indexes)} indexes")
    
    logger.success("✓ Database setup verified!")


async def main():
    """Main initialization function."""
    
    logger.info("=" * 60)
    logger.info("MTQuant PostgreSQL Database Initialization")
    logger.info("=" * 60)
    
    try:
        # Connect to database
        logger.info(f"Connecting to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
        conn = await asyncpg.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        logger.success("✓ Connected to PostgreSQL")
        
        # Create tables
        await create_tables(conn)
        
        # Create indexes
        await create_indexes(conn)
        
        # Create triggers
        await create_triggers(conn)
        
        # Insert sample data
        await insert_sample_data(conn)
        
        # Verify setup
        await verify_setup(conn)
        
        await conn.close()
        
        logger.info("=" * 60)
        logger.success("✅ PostgreSQL initialization completed successfully!")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Start Redis (if not running)")
        logger.info("2. Start QuestDB (if not running)")
        logger.info("3. Run backend: uvicorn api.main:app --reload")
        logger.info("4. Run frontend: cd frontend && npm run dev")
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

