# MT4 MCP Server Installation

## Overview
Uses [8nite/metatrader-4-mcp](https://github.com/8nite/metatrader-4-mcp)
- Node.js/TypeScript HTTP server
- File-based I/O with MT4 Expert Advisor
- MQL4 EA integration required

## Installation

### Prerequisites
- Node.js 18+ installed
- MetaTrader 4 terminal installed

1. Clone the MCP server:
```bash
cd mcp_servers/mt4
git clone https://github.com/8nite/metatrader-4-mcp.git server
cd server
npm install
```

2. Copy MT4 Expert Advisor:
- Locate: `server/mql4/MCPBridge.mq4`
- Copy to: `C:\Program Files\MetaTrader 4\MQL4\Experts\`
- Compile in MT4 MetaEditor

3. Start HTTP server:
```bash
npm start
# Runs on http://localhost:3000
```

## Configuration

Server endpoint: `http://localhost:3000`
File exchange directory: `C:\Users\%USERNAME%\AppData\Roaming\MetaQuotes\Terminal\<TERMINAL_ID>\MQL4\Files\`

## Usage

Start server:
```bash
cd mcp_servers/mt4/server
npm start
```

Test connection:
```bash
curl http://localhost:3000/health
```

## API Endpoints

- `POST /initialize` - Initialize MT4 connection
- `GET /market_data/:symbol` - Get OHLCV data
- `POST /order` - Place orders
- `GET /positions` - Get open positions
- `GET /account` - Get account information

## Requirements

- MT4 terminal must be running
- Expert Advisor must be attached to chart
- File permissions for MQL4/Files directory
