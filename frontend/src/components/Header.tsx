import { useQuery } from '@tanstack/react-query'
import { metricsAPI } from '@/services/api'
import { Activity, TrendingUp, TrendingDown } from 'lucide-react'
import clsx from 'clsx'

export default function Header() {
  const { data: metrics } = useQuery({
    queryKey: ['system-metrics'],
    queryFn: metricsAPI.getSystem,
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  return (
    <header className="h-16 bg-slate-800 border-b border-slate-700 flex items-center justify-between px-6">
      {/* System Status */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-green-500" />
          <span className="text-sm text-slate-300">System Online</span>
        </div>

        {metrics && (
          <>
            <div className="text-sm text-slate-300">
              <span className="text-slate-400">Agents:</span>{' '}
              <span className="font-semibold text-white">{metrics.active_agents}</span>
            </div>
            <div className="text-sm text-slate-300">
              <span className="text-slate-400">Positions:</span>{' '}
              <span className="font-semibold text-white">{metrics.open_positions}</span>
            </div>
            <div className="text-sm text-slate-300">
              <span className="text-slate-400">Orders Today:</span>{' '}
              <span className="font-semibold text-white">{metrics.orders_today}</span>
            </div>
          </>
        )}
      </div>

      {/* Time */}
      <div className="text-sm text-slate-400">
        {new Date().toLocaleString('en-US', {
          dateStyle: 'medium',
          timeStyle: 'short',
        })}
      </div>
    </header>
  )
}

