# MTQuant System Stop Script
# Zatrzymuje wszystkie komponenty systemu MTQuant

Write-Host "=" -ForegroundColor Cyan
Write-Host "🛑 MTQuant Trading System - Shutdown" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan
Write-Host ""

# Stop Backend (uvicorn/python)
Write-Host "🔧 Stopping Backend API..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -eq "python" -and $_.CommandLine -like "*uvicorn*"} | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process | Where-Object {$_.ProcessName -eq "uvicorn"} | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "✅ Backend stopped" -ForegroundColor Green

# Stop Frontend (node/npm)
Write-Host ""
Write-Host "🎨 Stopping Frontend..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -eq "node" -and $_.CommandLine -like "*vite*"} | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "✅ Frontend stopped" -ForegroundColor Green

# Stop QuestDB (optional)
Write-Host ""
Write-Host "📊 QuestDB is still running (use Ctrl+C in QuestDB window to stop)" -ForegroundColor Yellow

# Redis and PostgreSQL services remain running (system services)
Write-Host ""
Write-Host "🔴 Redis service is still running (Windows service)" -ForegroundColor Yellow
Write-Host "🐘 PostgreSQL service is still running (Windows service)" -ForegroundColor Yellow

Write-Host ""
Write-Host "=" -ForegroundColor Cyan
Write-Host "✅ MTQuant API and Frontend stopped" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan
Write-Host ""
Write-Host "ℹ️  Database services (Redis, PostgreSQL, QuestDB) are still running" -ForegroundColor White
Write-Host "   This is normal - they can run in the background." -ForegroundColor White
Write-Host ""

