"""
Test MT5 connection via MCP server.

This script tests if we can connect to MT5 using the MCP client.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mtquant.mcp_integration.adapters.mt5_adapter import MT5BrokerAdapter
from mtquant.mcp_integration.clients.mt5_mcp_client import BrokerConfig
from mtquant.utils.logger import get_logger

logger = get_logger(__name__)

# OANDA credentials
BROKER_ID = "oanda_mt5_demo"
MT5_ACCOUNT = 62675178
MT5_PASSWORD = "9RblZ8*K"
MT5_SERVER = "OANDATMS-MT5"


async def test_connection():
    """Test connection to MT5."""
    logger.info("🔌 Testing MT5 connection...")
    
    try:
        # Create broker config
        config = BrokerConfig(
            broker_id=BROKER_ID,
            account=MT5_ACCOUNT,
            password=MT5_PASSWORD,
            server=MT5_SERVER
        )
        
        # Create adapter
        adapter = MT5BrokerAdapter(
            broker_id=BROKER_ID,
            config=config
        )
        
        # Connect
        logger.info(f"Connecting to {MT5_SERVER} with account {MT5_ACCOUNT}...")
        await adapter.connect()
        logger.info("✅ Connected to MT5!")
        
        # Get account info
        logger.info("📊 Fetching account info...")
        account_info = await adapter.get_account_info()
        logger.info(f"Account Info: {account_info}")
        
        # Get symbols
        logger.info("📈 Fetching available symbols...")
        symbols = await adapter.get_symbols()
        logger.info(f"Found {len(symbols)} symbols")
        logger.info(f"First 10 symbols: {symbols[:10]}")
        
        # Get market data for XAUUSD
        if "XAUUSD" in symbols or "XAU_USD" in symbols:
            symbol = "XAUUSD" if "XAUUSD" in symbols else "XAU_USD"
            logger.info(f"📊 Fetching {symbol} market data...")
            
            market_data = await adapter.get_market_data(
                symbol=symbol,
                timeframe="1H",
                count=10
            )
            logger.info(f"{symbol} last 10 bars (1H):")
            logger.info(f"\n{market_data.tail()}")
        
        # Disconnect
        logger.info("🔌 Disconnecting...")
        await adapter.disconnect()
        logger.info("✅ Disconnected")
        
        logger.info("\n🎉 MT5 connection test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ MT5 connection test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)

