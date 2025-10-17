-- Broker connections table for persistence
CREATE TABLE IF NOT EXISTS broker_connections (
    broker_id VARCHAR(100) PRIMARY KEY,
    broker_type VARCHAR(10) NOT NULL CHECK (broker_type IN ('mt5', 'mt4')),
    account INTEGER NOT NULL,
    password_encrypted VARCHAR(500) NOT NULL,  -- TODO: encrypt in production!
    server VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_connected_at TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_broker_connections_active ON broker_connections(is_active);


