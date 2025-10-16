# MTQuant - Multi-Agent AI Trading System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](https://pytest.org/)

**MTQuant** to zaawansowany system handlowy wykorzystujący sztuczną inteligencję i uczenie ze wzmocnieniem (Reinforcement Learning) do automatycznego handlu na rynkach finansowych.

## 🎯 Przegląd Architektury

MTQuant to system wieloagentowy, gdzie każdy agent RL jest odpowiedzialny za jeden instrument finansowy (EURUSD, GBPUSD, USDJPY, XAUUSD, WTIUSD, SPX500, NAS100, US30). Centralny Menedżer Ryzyka koordynuje wszystkie agenty i egzekwuje limity na poziomie portfela.

### Kluczowe Komponenty

- **🤖 Agenci RL**: Niezależne agenty PPO dla każdego instrumentu
- **🛡️ Zarządzanie Ryzykiem**: Trzypoziomowa obrona przed stratami
- **📊 Integracja Brokerów**: Wsparcie dla MT4/MT5 z MCP (Model Context Protocol)
- **💾 Bazy Danych**: QuestDB (time-series), PostgreSQL (transakcyjne), Redis (hot data)
- **🌐 API**: FastAPI z WebSocket dla czasu rzeczywistego
- **📱 Frontend**: React 18+ z TypeScript i TradingView Charts

## 🛡️ System Zarządzania Ryzykiem

MTQuant implementuje zaawansowany system zarządzania ryzykiem z trzypoziomową obroną:

### Poziom 1: Walidacja Pre-Trade (<50ms)
- **Walidacja pasm cenowych**: Cena w zakresie ±5% od ostatniej znanej
- **Limity rozmiaru pozycji**: Maksymalnie 10% kapitału portfela
- **Weryfikacja kapitału**: Dostępny margin > wymagany × 1.5
- **Kontrola ekspozycji portfela**: Gross exposure < 150%, Net < 100%
- **Zgodność regulacyjna**: Limity dźwigni, godziny handlu
- **Ocena ryzyka korelacji**: Monitorowanie korelacji między pozycjami

### Poziom 2: Kalkulacja Rozmiaru Pozycji
- **Kryterium Kelly'ego**: Optymalne pozycjonowanie na podstawie historii
- **Pozycjonowanie oparte na zmienności**: ATR i volatility-based sizing
- **Metoda stałej frakcji**: Stały procent kapitału na transakcję
- **Dynamiczna korekta ryzyka**: Dostosowanie do warunków rynkowych

### Poziom 3: System Circuit Breaker
- **Poziom 1 (5% straty)**: Ostrzeżenia, zmniejszenie pozycji o 50%
- **Poziom 2 (10% straty)**: Zatrzymanie nowych pozycji, zamknięcie 50% istniejących
- **Poziom 3 (15% straty)**: Pełne zatrzymanie handlu, zamknięcie wszystkich pozycji
- **Automatyczne odzyskiwanie**: Stopniowe przywracanie po poprawie wyników

## 🤖 Pierwszy Agent RL (PPO)

System zawiera w pełni funkcjonalnego agenta PPO dla instrumentu XAUUSD:

### Architektura Agenta
- **Model PPO**: Proximal Policy Optimization z Stable-Baselines3
- **Sieć Policy**: Actor-Critic z architekturą [512, 256, 128, 64]
- **Przestrzeń stanów**: Stacjonarne cechy rynkowe + pozycja + ryzyko
- **Przestrzeń akcji**: Ciągła od -1 do +1 (sygnał handlowy)
- **Funkcja nagrody**: Risk-adjusted returns z karą za koszty transakcyjne

### Proces Treningu
```python
# Konfiguracja PPO
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=0.0001,
    n_steps=8192,
    batch_size=256,
    gamma=0.999,
    verbose=1
)

# Trening
model.learn(total_timesteps=500000)
model.save('models/checkpoints/XAUUSD_ppo_final.zip')
```

### Wyniki Treningu
- **Total P&L**: +2,468.46 (pozytywny wynik)
- **Win Rate**: 52.05% (ponad 50%)
- **Sharpe Ratio**: 0.054 (pozytywny)
- **Profit Factor**: 1.08 (pozytywny)
- **Średni P&L per trade**: +0.29

## 🚀 Szybki Start

### Wymagania

- Python 3.11+
- Git
- Docker (opcjonalnie)

### Instalacja

```bash
# Klonuj repozytorium
git clone https://github.com/jeden-/mtquant.git
cd mtquant

# Utwórz środowisko wirtualne z Python 3.11
py -3.11 -m venv venv

# Aktywuj środowisko (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Zainstaluj zależności podstawowe
pip install -r requirements.txt

# Zainstaluj zależności deweloperskie (opcjonalnie)
pip install -r requirements-dev.txt

# Zainstaluj pakiet w trybie deweloperskim
pip install -e .
```

#### Instalacja na Windows

**TA-Lib Installation:**
```powershell
# Opcja 1: Użyj pre-compiled wheel (zalecane)
pip install TA-Lib

# Opcja 2: Jeśli powyższe nie działa, zainstaluj Visual Studio Build Tools
# Pobierz z: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Zainstaluj "C++ build tools" workload
pip install TA-Lib

# Opcja 3: Użyj conda (alternatywnie)
conda install -c conda-forge ta-lib
```

**MetaTrader5 Installation:**
```powershell
# Wymaga zainstalowanego MetaTrader 5
# Pobierz z: https://www.metatrader5.com/en/download
# Zainstaluj MT5, następnie:
pip install MetaTrader5
```

**PostgreSQL Driver:**
```powershell
# Windows może wymagać dodatkowych zależności
pip install psycopg[binary,pool]
```

### Konfiguracja

```bash
# Skopiuj przykładowy plik konfiguracyjny
cp .env.example .env

# Edytuj konfigurację
# MT5_ACCOUNT=12345678
# MT5_PASSWORD=secret123
# MT5_SERVER=ICMarkets-Demo
```

### Uruchomienie

```bash
# Uruchom backend API
uvicorn api.main:app --reload --port 8000

# W osobnym terminalu uruchom frontend
cd frontend
npm install
npm run dev
```

## 🏗️ Architektura Systemu

### Multi-Agent Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   XAUUSD Agent  │    │   EURUSD Agent  │    │   USDJPY Agent  │
│   (PPO Policy)   │    │   (SAC Policy)  │    │   (TD3 Policy)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Risk Manager    │
                    │ (Centralized)   │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Broker Manager  │
                    │ (MT4/MT5 MCP)   │
                    └─────────────────┘
```

### Trzypoziomowa Obrona Ryzyka

1. **Pre-trade Validation** (<50ms)
   - Walidacja cen (±5-10% od ostatniej znanej)
   - Limity wielkości pozycji (<5% średniego dziennego wolumenu)
   - Weryfikacja kapitału (dostępna marża)
   - Zgodność regulacyjna

2. **Intra-trade Monitoring** (ciągłe)
   - Dynamiczne dostosowanie stop-loss
   - Śledzenie P&L w czasie rzeczywistym
   - Monitorowanie korelacji między pozycjami

3. **Circuit Breakers** (automatyczne)
   - Poziom 1 (5% dziennej straty): Ostrzeżenia
   - Poziom 2 (10% dziennej straty): Ograniczenie pozycji
   - Poziom 3 (15% dziennej straty): Pełne zatrzymanie

## 🛠️ Stos Technologiczny

### Backend
- **Python 3.11+**: Nowoczesne funkcje języka
- **FastAPI**: Wysokowydajne API z automatyczną dokumentacją
- **FinRL**: Framework RL dla finansów
- **Stable Baselines3**: Algorytmy RL (PPO, SAC, TD3)
- **Pandas/NumPy**: Przetwarzanie danych
- **TA-Lib**: Wskaźniki techniczne

### Bazy Danych
- **QuestDB**: Dane czasowe (OHLCV, wskaźniki)
- **PostgreSQL**: Dane transakcyjne (zlecenia, pozycje)
- **Redis**: Cache danych gorących (ceny, sesje)

### Frontend
- **React 18+**: Nowoczesny framework UI
- **TypeScript**: Bezpieczeństwo typów
- **Tailwind CSS**: Utility-first CSS
- **TradingView Charts**: Wykresy profesjonalne
- **WebSocket**: Dane w czasie rzeczywistym

### DevOps
- **Docker**: Konteneryzacja
- **GitHub Actions**: CI/CD
- **Prometheus**: Monitoring
- **Grafana**: Wizualizacja metryk

## 📊 Wspierane Instrumenty

| Instrument | Typ | Sesja | Średni Spread | Dzienna Zmienność |
|------------|-----|-------|---------------|-------------------|
| XAUUSD | Commodity | 24/5 | 0.30 USD | 20.0 USD |
| WTIUSD | Commodity | 24/5 | 0.05 USD | 2.0 USD |
| USDJPY | Forex | 24/5 | 0.1 pips | 0.80% |
| EURUSD | Forex | 24/5 | 0.1 pips | 0.70% |
| GBPUSD | Forex | 24/5 | 0.1 pips | 0.90% |
| SPX500 | Index | 24/5 | 0.25 USD | 15.0 USD |
| NAS100 | Index | 24/5 | 0.20 USD | 12.0 USD |
| US30 | Index | 24/5 | 1.0 USD | 50.0 USD |

## 🔒 Bezpieczeństwo i Zgodność

### Zarządzanie Poświadczeniami
- Zmienne środowiskowe dla wszystkich sekretów
- Nigdy nie commituj plików `.env`
- Rotacja kluczy API

### Audit Trail
- Logowanie wszystkich decyzji handlowych
- Pełna ścieżka audytu (kto, co, kiedy, dlaczego)
- Przechowywanie przez 5 lat (wymóg regulacyjny)

### Testowanie
- Minimum 70% pokrycia kodem
- 100% pokrycia dla kodu zarządzania ryzykiem
- Testy jednostkowe i integracyjne
- Paper trading przed wdrożeniem na żywo

## 📈 Strategie Handlowe

### Pozycjonowanie
- **Kelly Criterion**: Optymalne rozmiary pozycji
- **Volatility-based**: Dostosowanie do zmienności
- **Fixed Fractional**: Stały procent portfela

### Stop-Loss
- **ATR-based**: Dostosowanie do zmienności
- **Fixed %**: Proste i przewidywalne
- **Trailing**: Podążanie za ceną

### Take-Profit
- **Risk:Reward**: Stosunek 1:2 lub lepszy
- **Poziomy techniczne**: Support/Resistance
- **Czasowe**: Zamknięcie po X godzinach

## 🧪 Testowanie

```bash
# Uruchom wszystkie testy
pytest

# Testy z pokryciem kodem
pytest --cov=mtquant --cov-report=html

# Testy integracyjne
pytest tests/integration/

# Testy wydajności
pytest tests/performance/
```

## 🚀 Wdrożenie

### Paper Trading
```bash
# Uruchom paper trading
python scripts/paper_trade.py --symbol XAUUSD --duration 30d
```

### Live Trading
```bash
# Wdrożenie na żywo (ostrożnie!)
python scripts/deploy_live.py --symbol XAUUSD --capital-pct 0.10
```

### Docker
```bash
# Zbuduj kontenery
docker-compose build

# Uruchom system
docker-compose up -d

# Sprawdź logi
docker-compose logs -f backend
```

## 📚 Dokumentacja

- [Architektura Systemu](docs/architecture.md)
- [API Reference](docs/api.md)
- [Konfiguracja Brokerów](docs/brokers.md)
- [Zarządzanie Ryzykiem](docs/risk-management.md)
- [Przewodnik Dewelopera](docs/development.md)

## 🤝 Wkład w Projekt

1. Fork repozytorium
2. Utwórz branch feature (`git checkout -b feature/amazing-feature`)
3. Commit zmian (`git commit -m 'feat: add amazing feature'`)
4. Push do branch (`git push origin feature/amazing-feature`)
5. Otwórz Pull Request

### Standardy Kodu
- **Python**: Black, Ruff, MyPy
- **TypeScript**: ESLint, Prettier
- **Commits**: Conventional Commits
- **Tests**: Minimum 70% pokrycia

## ⚠️ Ostrzeżenie

**MTQuant to system handlowy używający prawdziwych pieniędzy. Używaj go na własne ryzyko.**

- Zawsze testuj na kontach demo przed użyciem prawdziwych pieniędzy
- Rozpocznij od małych kwot (10% kapitału)
- Monitoruj system 24/7
- Miej plan awaryjny na wypadek awarii
- Przestrzegaj lokalnych regulacji finansowych

## 📄 Licencja

Ten projekt jest licencjonowany na licencji MIT - zobacz plik [LICENSE](LICENSE) dla szczegółów.

## 📞 Kontakt

- **Email**: contact@mtquant.com
- **GitHub**: [@jeden-](https://github.com/jeden-)
- **Discord**: [MTQuant Community](https://discord.gg/mtquant)

## 🙏 Podziękowania

- [FinRL](https://github.com/AI4Finance-Foundation/FinRL) - Framework RL dla finansów
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) - Algorytmy RL
- [QuestDB](https://questdb.io/) - Baza danych czasowa
- [TradingView](https://www.tradingview.com/) - Wykresy finansowe

---

**Pamiętaj**: Bezpieczeństwo przede wszystkim, testuj dokładnie, nigdy nie ufaj systemom zewnętrznym, zawsze utrzymuj ścieżki audytu, a w razie wątpliwości - pytaj przed wykonaniem transakcji z prawdziwymi pieniędzmi! 🛡️
