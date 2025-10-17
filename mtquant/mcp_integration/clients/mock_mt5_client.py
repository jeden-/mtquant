"""Mock MT5 client for testing broker persistence without MCP issues."""
import asyncio
from typing import Dict, Any
from datetime import datetime

from mtquant.utils.logger import get_logger


class MockMT5Client:
    """Mock MT5 client that always works for testing."""
    
    def __init__(self, broker_id: str, config: Dict[str, Any]):
        self.broker_id = broker_id
        self.config = config
        self.logger = get_logger(__name__)
        self._connected = False
        self._last_health_check = None
    
    async def connect(self) -> bool:
        """Mock connection - always succeeds."""
        try:
            self.logger.info(f"🔧 MOCK: Connecting to MT5 for {self.broker_id}")
            
            # Simulate connection delay
            await asyncio.sleep(0.1)
            
            self._connected = True
            self.logger.info(f"✅ MOCK: Connected to MT5: {self.broker_id}")
            return True
                
        except Exception as e:
            self.logger.exception("MOCK MT5 connection error")
            return False
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Mock account info."""
        return {
            "balance": 100000.0,
            "equity": 100000.0,
            "margin": 0.0,
            "free_margin": 100000.0,
            "profit": 0.0,
            "leverage": 100,
            "login": self.config.get('account', 12345),
            "server": self.config.get('server', 'MOCK-SERVER')
        }
    
    async def health_check(self) -> bool:
        """Mock health check."""
        if not self._connected:
            return False
        
        self._last_health_check = datetime.utcnow()
        return True
    
    async def disconnect(self) -> None:
        """Mock disconnect."""
        self._connected = False
        self.logger.info(f"🔧 MOCK: Disconnected from MT5: {self.broker_id}")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

