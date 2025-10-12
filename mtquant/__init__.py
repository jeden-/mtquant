"""
MTQuant - Multi-Agent AI Trading System using Reinforcement Learning

A production-grade trading system that combines RL agents with centralized risk management
for safe, automated trading across multiple instruments (XAUUSD, BTCUSD, USDJPY, EURUSD).

Architecture:
- Multi-Agent: Independent RL agents per instrument
- Centralized Risk: Risk Manager coordinates all agents
- Hybrid Design: RL signals + rule-based position sizing
- Production Focus: Safety-first, regulatory-compliant

Core Technologies:
- Backend: Python 3.11+, FastAPI, FinRL, Stable Baselines3
- Frontend: React 18+, TypeScript, Tailwind CSS
- Databases: QuestDB (time-series), PostgreSQL (transactional), Redis (hot data)
- Broker Integration: MetaTrader 4/5 via MCP servers
"""

__version__ = "0.1.0"
__author__ = "MTQuant Team"
__email__ = "contact@mtquant.com"
__license__ = "MIT"

# Core modules
from . import agents
from . import mcp_integration
from . import risk_management
from . import data
from . import utils

__all__ = [
    "agents",
    "mcp_integration", 
    "risk_management",
    "data",
    "utils",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]
