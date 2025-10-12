# MT5 MCP Server Installation

## Overview
Uses [Qoyyuum/mcp-metatrader5-server](https://github.com/Qoyyuum/mcp-metatrader5-server)
- Python-based FastMCP server
- Direct MetaTrader5 package integration
- Tools: initialize, login, shutdown, get_symbols, copy_rates_from_pos, order_send, etc.

## Installation

1. Clone the MCP server:
```bash
cd mcp_servers/mt5
git clone https://github.com/Qoyyuum/mcp-metatrader5-server.git server
cd server
```

2. Install with uv:
```bash
uv run fastmcp install src/mcp_metatrader5_server/server.py
```

3. Test installation:
```bash
uv run mt5mcp dev
# Should start server on http://127.0.0.1:8000
```

## Configuration

Server path for MTQuant: `C:\FULL_PATH\mtquant\mcp_servers\mt5\server`

Environment variables (in .env):
- MT5_DEMO_ACCOUNT
- MT5_DEMO_PASSWORD
- MT5_DEMO_SERVER

## Usage

Start server:
```bash
cd mcp_servers/mt5/server
uv run mt5mcp dev
```

Test connection:
```bash
curl http://localhost:8000/health
```

## Tools Available

- `initialize` - Initialize MT5 terminal
- `login` - Login to MT5 account
- `shutdown` - Shutdown MT5 connection
- `get_symbols` - Get available symbols
- `copy_rates_from_pos` - Get OHLCV data
- `order_send` - Place orders
- `positions_get` - Get open positions
- `account_info` - Get account information
