import axios from 'axios'
import type { Agent, PortfolioSummary, Position, Order, SystemMetrics } from '@/types'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Agent API
export const agentAPI = {
  getAll: async (): Promise<Agent[]> => {
    const { data } = await api.get('/agents')
    return data
  },

  getById: async (id: string): Promise<Agent> => {
    const { data } = await api.get(`/agents/${id}`)
    return data
  },

  start: async (id: string): Promise<void> => {
    await api.post(`/agents/${id}/start`)
  },

  stop: async (id: string): Promise<void> => {
    await api.post(`/agents/${id}/stop`)
  },

  pause: async (id: string): Promise<void> => {
    await api.post(`/agents/${id}/pause`)
  },

  resume: async (id: string): Promise<void> => {
    await api.post(`/agents/${id}/resume`)
  },
}

// Portfolio API
export const portfolioAPI = {
  getSummary: async (): Promise<PortfolioSummary> => {
    const { data } = await api.get('/portfolio/summary')
    return data
  },

  getPositions: async (): Promise<Position[]> => {
    const { data } = await api.get('/portfolio/positions')
    return data
  },

  getRiskMetrics: async (): Promise<any> => {
    const { data } = await api.get('/portfolio/risk')
    return data
  },
}

// Orders API
export const ordersAPI = {
  getAll: async (limit = 100): Promise<Order[]> => {
    const { data } = await api.get('/orders', { params: { limit } })
    return data
  },

  place: async (order: Partial<Order>): Promise<Order> => {
    const { data } = await api.post('/orders', order)
    return data
  },

  cancel: async (orderId: string): Promise<void> => {
    await api.delete(`/orders/${orderId}`)
  },
}

// Metrics API
export const metricsAPI = {
  getSystem: async (): Promise<SystemMetrics> => {
    const { data } = await api.get('/metrics/system')
    return data
  },

  getAgentMetrics: async (agentId: string, periodDays = 30): Promise<any> => {
    const { data } = await api.get(`/metrics/agents/${agentId}`, {
      params: { period_days: periodDays },
    })
    return data
  },
}

export default api


