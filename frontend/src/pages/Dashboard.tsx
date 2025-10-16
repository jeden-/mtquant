import { useQuery } from '@tanstack/react-query'
import { portfolioAPI, agentAPI } from '@/services/api'
import { TrendingUp, TrendingDown, DollarSign, Activity } from 'lucide-react'
import clsx from 'clsx'

export default function Dashboard() {
  const { data: portfolio, isLoading: portfolioLoading, error: portfolioError } = useQuery({
    queryKey: ['portfolio-summary'],
    queryFn: portfolioAPI.getSummary,
    refetchInterval: 5000,
  })

  const { data: agents, isLoading: agentsLoading, error: agentsError } = useQuery({
    queryKey: ['agents'],
    queryFn: agentAPI.getAll,
    refetchInterval: 10000,
  })

  const activeAgents = agents?.filter((a) => a.status === 'live').length || 0

  // Show loading state
  if (portfolioLoading || agentsLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-white text-xl">Loading...</div>
      </div>
    )
  }

  // Show error state
  if (portfolioError || agentsError) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-red-500 text-xl">
          Error loading dashboard: {portfolioError?.message || agentsError?.message}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">Dashboard</h1>
        <p className="text-slate-400 mt-1">
          Overview of your trading system performance
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Equity */}
        <StatCard
          title="Total Equity"
          value={portfolio?.total_equity || 0}
          format="currency"
          icon={DollarSign}
          iconColor="text-green-500"
        />

        {/* Daily P&L */}
        <StatCard
          title="Daily P&L"
          value={portfolio?.daily_pnl || 0}
          format="currency"
          change={portfolio?.daily_pnl_pct}
          icon={portfolio && portfolio.daily_pnl >= 0 ? TrendingUp : TrendingDown}
          iconColor={portfolio && portfolio.daily_pnl >= 0 ? 'text-green-500' : 'text-red-500'}
        />

        {/* Active Agents */}
        <StatCard
          title="Active Agents"
          value={activeAgents}
          format="number"
          icon={Activity}
          iconColor="text-blue-500"
        />

        {/* Unrealized P&L */}
        <StatCard
          title="Unrealized P&L"
          value={portfolio?.total_unrealized_pnl || 0}
          format="currency"
          icon={portfolio && portfolio.total_unrealized_pnl >= 0 ? TrendingUp : TrendingDown}
          iconColor={
            portfolio && portfolio.total_unrealized_pnl >= 0 ? 'text-green-500' : 'text-red-500'
          }
        />
      </div>

      {/* Agents Overview */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Active Agents</h2>
        <div className="space-y-3">
          {agents?.slice(0, 5).map((agent) => (
            <div
              key={agent.id}
              className="flex items-center justify-between p-4 bg-slate-900 rounded-lg"
            >
              <div className="flex items-center gap-4">
                <div
                  className={clsx(
                    'w-3 h-3 rounded-full',
                    agent.status === 'live' && 'bg-green-500',
                    agent.status === 'paper' && 'bg-yellow-500',
                    agent.status === 'paused' && 'bg-gray-500',
                    agent.status === 'error' && 'bg-red-500'
                  )}
                />
                <div>
                  <div className="text-white font-medium">{agent.symbol}</div>
                  <div className="text-sm text-slate-400">{agent.id}</div>
                </div>
              </div>
              <div className="text-right">
                <div
                  className={clsx(
                    'text-lg font-semibold',
                    agent.unrealized_pnl >= 0 ? 'text-green-500' : 'text-red-500'
                  )}
                >
                  {agent.unrealized_pnl >= 0 ? '+' : ''}
                  ${agent.unrealized_pnl.toFixed(2)}
                </div>
                <div className="text-sm text-slate-400">{agent.today_trades} trades today</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// Stat Card Component
interface StatCardProps {
  title: string
  value: number
  format: 'currency' | 'number' | 'percent'
  change?: number
  icon: React.ComponentType<{ className?: string }>
  iconColor: string
}

function StatCard({ title, value, format, change, icon: Icon, iconColor }: StatCardProps) {
  const formattedValue =
    format === 'currency'
      ? `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
      : format === 'percent'
        ? `${value.toFixed(2)}%`
        : value.toLocaleString()

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <span className="text-slate-400 text-sm font-medium">{title}</span>
        <Icon className={clsx('w-5 h-5', iconColor)} />
      </div>
      <div className="text-2xl font-bold text-white">{formattedValue}</div>
      {change !== undefined && (
        <div
          className={clsx(
            'text-sm font-medium mt-2',
            change >= 0 ? 'text-green-500' : 'text-red-500'
          )}
        >
          {change >= 0 ? '+' : ''}
          {change.toFixed(2)}%
        </div>
      )}
    </div>
  )
}

