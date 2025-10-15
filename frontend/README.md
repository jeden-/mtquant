# MTQuant Frontend

Modern React + TypeScript frontend for the MTQuant multi-agent trading system.

## 🚀 Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **TanStack Query** - Data fetching
- **React Router** - Routing
- **Lightweight Charts** - Trading charts (TradingView)
- **Zustand** - State management
- **Lucide React** - Icons

## 📦 Installation

```bash
npm install
```

## 🛠️ Development

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

## 🏗️ Build

```bash
npm run build
```

## 📁 Project Structure

```
src/
├── components/     # Reusable UI components
├── pages/          # Page components
├── services/       # API clients
├── hooks/          # Custom hooks
├── store/          # State management
├── types/          # TypeScript types
└── main.tsx        # Entry point
```

## 🔌 API Integration

Frontend connects to backend at `http://localhost:8000/api`

WebSocket endpoints:
- `/ws/portfolio` - Portfolio updates
- `/ws/orders` - Order updates
- `/ws/agents` - Agent status
- `/ws/market` - Market data

## 🎨 Features

- ✅ Real-time dashboard
- ✅ Agent monitoring
- ✅ Portfolio overview
- ✅ Order management
- ⏳ Analytics (coming soon)
- ⏳ TradingView charts (coming soon)

## 📝 License

MIT

