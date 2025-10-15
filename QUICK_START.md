# 🚀 MTQuant - Quick Start Guide

## Uruchomienie systemu

### Metoda 1: Automatyczny start (ZALECANE)

Uruchom skrypt startowy w PowerShell:

```powershell
.\start.ps1
```

To uruchomi wszystkie komponenty:
- ✅ QuestDB (time-series database)
- ✅ Redis (cache)
- ✅ PostgreSQL (transactional database)
- ✅ Backend API (FastAPI)
- ✅ Frontend (React + Vite)

### Metoda 2: Ręczny start

#### 1. Uruchom bazy danych:

**QuestDB:**
```powershell
cd questdb-9.1.0-rt-windows-x86-64\bin
.\questdb.exe
```

**Redis** (usługa Windows - powinna działać automatycznie)
```powershell
# Sprawdź status
Get-Service Redis

# Jeśli nie działa:
Start-Service Redis
```

**PostgreSQL** (usługa Windows - powinna działać automatycznie)
```powershell
# Sprawdź status
Get-Service | Where-Object {$_.DisplayName -like "*postgres*"}
```

#### 2. Uruchom Backend API:

```powershell
uvicorn api.main:app --reload --port 8000
```

#### 3. Uruchom Frontend:

```powershell
cd frontend
npm run dev
```

## Zatrzymanie systemu

### Automatyczne zatrzymanie:

```powershell
.\stop.ps1
```

### Ręczne zatrzymanie:

- **Backend**: Naciśnij `Ctrl+C` w terminalu backendu
- **Frontend**: Naciśnij `Ctrl+C` w terminalu frontendu
- **QuestDB**: Zamknij okno konsoli QuestDB
- **Redis/PostgreSQL**: Pozostaną uruchomione jako usługi Windows (to OK)

## Dostęp do systemu

Po uruchomieniu:

| Komponent | URL | Opis |
|-----------|-----|------|
| **Frontend** | http://localhost:5173 | Główny interfejs użytkownika |
| **Backend API** | http://localhost:8000 | REST API |
| **API Docs** | http://localhost:8000/api/docs | Swagger UI - dokumentacja API |
| **QuestDB Console** | http://localhost:9000 | Konsola SQL dla QuestDB |
| **Health Check** | http://localhost:8000/health | Status systemu |

## Pierwsze kroki

### 1. Sprawdź status systemu

Otwórz: http://localhost:8000/health

Powinieneś zobaczyć:
```json
{
  "status": "healthy",
  "timestamp": 1234567890.123
}
```

### 2. Eksploruj API

Otwórz: http://localhost:8000/api/docs

Tutaj znajdziesz:
- Wszystkie dostępne endpointy
- Możliwość testowania API
- Szczegółową dokumentację

### 3. Otwórz Frontend

Otwórz: http://localhost:5173

Zobaczysz:
- **Dashboard** - Przegląd systemu
- **Agents** - Zarządzanie agentami RL
- **Portfolio** - Twoje pozycje i P&L
- **Orders** - Historia transakcji
- **Analytics** - Wykresy i statystyki
- **Settings** - Konfiguracja systemu

### 4. Uruchom trening agenta

```powershell
python scripts/run_phase1_training.py
```

## Troubleshooting

### Backend nie uruchamia się

1. Sprawdź czy wszystkie bazy działają:
```powershell
# Redis
redis-cli ping

# PostgreSQL
Get-Service | Where-Object {$_.DisplayName -like "*postgres*"}
```

2. Sprawdź logi w terminalu backendu

### Frontend nie uruchamia się

1. Sprawdź czy backend działa:
```powershell
curl http://localhost:8000/health
```

2. Sprawdź czy node_modules są zainstalowane:
```powershell
cd frontend
npm install
```

### Port zajęty

Jeśli port 8000 lub 5173 jest zajęty:

**Backend (zmień port):**
```powershell
uvicorn api.main:app --reload --port 8001
```

**Frontend (zmień port w vite.config.ts):**
```typescript
server: {
  port: 5174  // Zmień na inny port
}
```

## Konfiguracja

### Zmienne środowiskowe

Edytuj `.env`:

```bash
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mtquantum
POSTGRES_USER=postgres
POSTGRES_PASSWORD=MARiusz@!2025

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# QuestDB
QUESTDB_HOST=localhost
QUESTDB_PORT=8812
```

## Dokumentacja

- **Architektura**: `docs/architecture.md`
- **Risk Management**: `docs/risk-management.md`
- **RL Agents**: `docs/rl-agents.md`
- **Sprint Docs**: `docs/sprint_03_doc.md`

## Wsparcie

W razie problemów:
1. Sprawdź logi w terminalach
2. Sprawdź `logs/` folder
3. Otwórz issue na GitHub

---

**Powodzenia w tradingu! 🚀📈**

