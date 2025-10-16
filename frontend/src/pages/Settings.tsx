import { useState, useEffect } from 'react'
import { CheckCircle, XCircle, Loader2, AlertCircle } from 'lucide-react'

interface BrokerConnection {
  id: string
  name: string
  type: 'MT5' | 'MT4'
  status: 'connected' | 'disconnected' | 'connecting' | 'error'
  account?: string
  server?: string
}

export default function Settings() {
  const [brokers, setBrokers] = useState<BrokerConnection[]>([])
  const [showAddBroker, setShowAddBroker] = useState(false)
  
  // Load connected brokers on mount
  useEffect(() => {
    loadConnectedBrokers()
  }, [])
  
  const loadConnectedBrokers = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/brokers/list')
      const data = await response.json()
      
      if (data.brokers) {
        const formattedBrokers: BrokerConnection[] = data.brokers.map((b: any) => ({
          id: b.broker_id,
          name: `${b.broker_id}`,
          type: b.broker_id.startsWith('mt5') ? 'MT5' : 'MT4',
          status: b.connected ? 'connected' : 'error',
          account: b.account_info?.login?.toString(),
          server: b.account_info?.server
        }))
        setBrokers(formattedBrokers)
      }
    } catch (err) {
      console.error('Failed to load brokers:', err)
    }
  }
  
  // Form state
  const [brokerType, setBrokerType] = useState<'MT5' | 'MT4'>('MT5')
  const [account, setAccount] = useState('')
  const [password, setPassword] = useState('')
  const [server, setServer] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleConnectBroker = async () => {
    setLoading(true)
    setError(null)
    
    try {
      // Add broker to list with "connecting" status
      const newBroker: BrokerConnection = {
        id: `broker_${Date.now()}`,
        name: `${brokerType} - ${account}`,
        type: brokerType,
        status: 'connecting',
        account,
        server
      }
      
      setBrokers([...brokers, newBroker])
      
      // Call backend API to connect
      const response = await fetch('http://localhost:8000/api/brokers/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          broker_type: brokerType.toLowerCase(),
          account: parseInt(account),
          password,
          server
        })
      })
      
      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to connect')
      }
      
      const data = await response.json()
      
      // Reload brokers from backend to get accurate info
      await loadConnectedBrokers()
      
      // Reset form
      setAccount('')
      setPassword('')
      setServer('')
      setShowAddBroker(false)
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connection failed')
      
      // Update broker status to error
      setBrokers(prev => prev.map(b => 
        b.status === 'connecting' 
          ? { ...b, status: 'error' as const }
          : b
      ))
    } finally {
      setLoading(false)
    }
  }

  const handleDisconnect = async (brokerId: string) => {
    try {
      await fetch(`http://localhost:8000/api/brokers/${brokerId}/disconnect`, {
        method: 'POST'
      })
      
      // Reload brokers from backend
      await loadConnectedBrokers()
    } catch (err) {
      console.error('Failed to disconnect:', err)
    }
  }

  const getStatusIcon = (status: BrokerConnection['status']) => {
    switch (status) {
      case 'connected':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'disconnected':
        return <XCircle className="h-5 w-5 text-slate-500" />
      case 'connecting':
        return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-500" />
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">Settings</h1>
      </div>

      {/* Broker Connections Section */}
      <div className="bg-slate-800 rounded-lg p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-white">Broker Connections</h2>
          <button
            onClick={() => setShowAddBroker(!showAddBroker)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            {showAddBroker ? 'Cancel' : '+ Add Broker'}
          </button>
        </div>

        {/* Add Broker Form */}
        {showAddBroker && (
          <div className="bg-slate-700 rounded-lg p-6 space-y-4">
            <h3 className="text-lg font-medium text-white mb-4">Connect New Broker</h3>
            
            {error && (
              <div className="bg-red-500/10 border border-red-500 rounded-lg p-4 flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
                <div className="text-red-200">{error}</div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Broker Type
                </label>
                <select
                  value={brokerType}
                  onChange={(e) => setBrokerType(e.target.value as 'MT5' | 'MT4')}
                  className="w-full bg-slate-600 border border-slate-500 rounded-lg px-4 py-2 text-white"
                >
                  <option value="MT5">MetaTrader 5</option>
                  <option value="MT4">MetaTrader 4</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Account Number
                </label>
                <input
                  type="text"
                  value={account}
                  onChange={(e) => setAccount(e.target.value)}
                  placeholder="12345678"
                  className="w-full bg-slate-600 border border-slate-500 rounded-lg px-4 py-2 text-white placeholder-slate-400"
                />
              </div>

                     <div>
                       <label className="block text-sm font-medium text-slate-300 mb-2">
                         Password
                       </label>
                       <input
                         type="password"
                         value={password}
                         onChange={(e) => setPassword(e.target.value)}
                         placeholder="9Rb!Z8*K"
                         className="w-full bg-slate-600 border border-slate-500 rounded-lg px-4 py-2 text-white placeholder-slate-400"
                       />
                     </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Server
                </label>
                <input
                  type="text"
                  value={server}
                  onChange={(e) => setServer(e.target.value)}
                  placeholder="ICMarkets-Demo"
                  className="w-full bg-slate-600 border border-slate-500 rounded-lg px-4 py-2 text-white placeholder-slate-400"
                />
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowAddBroker(false)}
                className="px-6 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleConnectBroker}
                disabled={loading || !account || !password || !server}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                {loading ? 'Connecting...' : 'Connect'}
              </button>
            </div>
          </div>
        )}

        {/* Connected Brokers List */}
        {brokers.length > 0 ? (
          <div className="space-y-3">
            {brokers.map((broker) => (
              <div
                key={broker.id}
                className="bg-slate-700 rounded-lg p-4 flex items-center justify-between"
              >
                <div className="flex items-center gap-4">
                  {getStatusIcon(broker.status)}
                  <div>
                    <div className="text-white font-medium">{broker.name}</div>
                    <div className="text-sm text-slate-400">
                      {broker.server} • Account: {broker.account}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center gap-3">
                  <span className={`text-sm px-3 py-1 rounded-full ${
                    broker.status === 'connected' 
                      ? 'bg-green-500/20 text-green-400'
                      : broker.status === 'error'
                      ? 'bg-red-500/20 text-red-400'
                      : 'bg-slate-600 text-slate-400'
                  }`}>
                    {broker.status}
                  </span>
                  
                  {broker.status === 'connected' && (
                    <button
                      onClick={() => handleDisconnect(broker.id)}
                      className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                    >
                      Disconnect
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12 text-slate-400">
            <p>No brokers connected</p>
            <p className="text-sm mt-2">Click "Add Broker" to connect MT5 or MT4</p>
          </div>
        )}
      </div>

      {/* System Settings */}
      <div className="bg-slate-800 rounded-lg p-6 space-y-4">
        <h2 className="text-xl font-semibold text-white">System Settings</h2>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between py-3 border-b border-slate-700">
            <div>
              <div className="text-white font-medium">Auto-start Agents</div>
              <div className="text-sm text-slate-400">Automatically start trained agents on system boot</div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" />
              <div className="w-11 h-6 bg-slate-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
            </label>
          </div>

          <div className="flex items-center justify-between py-3 border-b border-slate-700">
            <div>
              <div className="text-white font-medium">Paper Trading Mode</div>
              <div className="text-sm text-slate-400">Trade with virtual money (recommended for testing)</div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" defaultChecked />
              <div className="w-11 h-6 bg-slate-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
            </label>
          </div>

          <div className="flex items-center justify-between py-3">
            <div>
              <div className="text-white font-medium">Risk Alerts</div>
              <div className="text-sm text-slate-400">Get notified when risk limits are approached</div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" defaultChecked />
              <div className="w-11 h-6 bg-slate-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
            </label>
          </div>
        </div>
      </div>

      {/* Database Status */}
      <div className="bg-slate-800 rounded-lg p-6 space-y-4">
        <h2 className="text-xl font-semibold text-white">Database Status</h2>
        
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-slate-700 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              <div className="text-white font-medium">PostgreSQL</div>
            </div>
            <div className="text-sm text-slate-400">localhost:5432</div>
          </div>

          <div className="bg-slate-700 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              <div className="text-white font-medium">QuestDB</div>
            </div>
            <div className="text-sm text-slate-400">localhost:8812</div>
          </div>

          <div className="bg-slate-700 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              <div className="text-white font-medium">Redis</div>
            </div>
            <div className="text-sm text-slate-400">localhost:6379</div>
          </div>
        </div>
      </div>
    </div>
  )
}
