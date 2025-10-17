// Agent types
export interface Agent {
  id: string
  symbol: string
  status: 'training' | 'paper' | 'live' | 'paused' | 'error'
  agent_type: string
  current_position: Position | null
  unrealized_pnl: number
  today_trades: number
  metrics: AgentMetrics
}

export interface AgentMetrics {
  sharpe_ratio: number
  win_rate: number
  total_trades: number
  total_pnl: number
  max_drawdown: number
}

// Position types
export interface Position {
  position_id: string
  agent_id: string
  symbol: string
  side: 'long' | 'short'
  quantity: number
  entry_price: number
  current_price: number
  unrealized_pnl: number
  unrealized_pnl_pct: number
  opened_at: string
  stop_loss?: number
  take_profit?: number
}

// Portfolio types
export interface PortfolioSummary {
  total_equity: number
  cash_balance: number
  total_positions_value: number
  total_unrealized_pnl: number
  total_realized_pnl: number
  daily_pnl: number
  daily_pnl_pct: number
  margin_used: number
  margin_available: number
  margin_usage_pct: number
}

// Order types
export interface Order {
  order_id: string
  agent_id: string
  symbol: string
  side: 'buy' | 'sell'
  order_type: 'market' | 'limit' | 'stop'
  quantity: number
  price?: number
  status: 'pending' | 'filled' | 'cancelled' | 'rejected'
  created_at: string
  filled_at?: string
}

// Metrics types
export interface SystemMetrics {
  timestamp: string
  cpu_usage_percent: number
  memory_usage_percent: number
  disk_usage_percent: number
  active_agents: number
  open_positions: number
  orders_today: number
  websocket_connections: number
}

// WebSocket message types
export interface WSMessage {
  type: 'portfolio_update' | 'order_update' | 'agent_update' | 'market_update' | 'heartbeat' | 'pong'
  data?: any
  timestamp: string
}



