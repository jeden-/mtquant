import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Bot,
  Briefcase,
  ShoppingCart,
  BarChart3,
  Settings,
} from 'lucide-react'
import clsx from 'clsx'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Agents', href: '/agents', icon: Bot },
  { name: 'Portfolio', href: '/portfolio', icon: Briefcase },
  { name: 'Orders', href: '/orders', icon: ShoppingCart },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  { name: 'Settings', href: '/settings', icon: Settings },
]

export default function Sidebar() {
  return (
    <div className="w-64 bg-slate-800 border-r border-slate-700">
      {/* Logo */}
      <div className="h-16 flex items-center justify-center border-b border-slate-700">
        <h1 className="text-2xl font-bold text-white">MTQuant</h1>
      </div>

      {/* Navigation */}
      <nav className="p-4 space-y-1">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            end={item.href === '/'}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-4 py-3 rounded-lg transition-colors',
                isActive
                  ? 'bg-primary-600 text-white'
                  : 'text-slate-300 hover:bg-slate-700 hover:text-white'
              )
            }
          >
            <item.icon className="w-5 h-5" />
            <span className="font-medium">{item.name}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="absolute bottom-0 w-64 p-4 border-t border-slate-700">
        <div className="text-xs text-slate-400 text-center">
          v0.1.0 | © 2025 MTQuant
        </div>
      </div>
    </div>
  )
}



