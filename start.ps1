# MTQuant System Startup Script
# Uruchamia wszystkie komponenty systemu MTQuant

Write-Host "=" -ForegroundColor Cyan
Write-Host "🚀 MTQuant Trading System - Startup" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan
Write-Host ""

# Check if QuestDB is running
$questdbProcess = Get-Process | Where-Object {$_.ProcessName -like "*java*" -and $_.Path -like "*questdb*"}
if (-not $questdbProcess) {
    Write-Host "📊 Starting QuestDB..." -ForegroundColor Yellow
    Start-Process -FilePath ".\questdb-9.1.0-rt-windows-x86-64\bin\questdb.exe" -WindowStyle Minimized
    Start-Sleep -Seconds 3
    Write-Host "✅ QuestDB started" -ForegroundColor Green
} else {
    Write-Host "✅ QuestDB already running" -ForegroundColor Green
}

# Check if Redis is running
$redisService = Get-Service | Where-Object {$_.Name -eq "Redis"}
if ($redisService.Status -ne "Running") {
    Write-Host "🔴 Starting Redis service..." -ForegroundColor Yellow
    Start-Service Redis
    Write-Host "✅ Redis started" -ForegroundColor Green
} else {
    Write-Host "✅ Redis already running" -ForegroundColor Green
}

# Check if PostgreSQL is running
$postgresService = Get-Service | Where-Object {$_.DisplayName -like "*postgres*"}
if ($postgresService.Status -ne "Running") {
    Write-Host "🐘 Starting PostgreSQL service..." -ForegroundColor Yellow
    Start-Service $postgresService.Name
    Write-Host "✅ PostgreSQL started" -ForegroundColor Green
} else {
    Write-Host "✅ PostgreSQL already running" -ForegroundColor Green
}

Write-Host ""
Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start Backend API
Write-Host ""
Write-Host "🔧 Starting Backend API..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; uvicorn api.main:app --reload --port 8000" -WindowStyle Normal
Start-Sleep -Seconds 3
Write-Host "✅ Backend API starting (http://localhost:8000)" -ForegroundColor Green

# Start Frontend
Write-Host ""
Write-Host "🎨 Starting Frontend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\frontend'; npm run dev" -WindowStyle Normal
Start-Sleep -Seconds 2
Write-Host "✅ Frontend starting (http://localhost:5173)" -ForegroundColor Green

Write-Host ""
Write-Host "=" -ForegroundColor Cyan
Write-Host "🎉 MTQuant System Started Successfully!" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Services:" -ForegroundColor White
Write-Host "  • Backend API:  http://localhost:8000" -ForegroundColor White
Write-Host "  • API Docs:     http://localhost:8000/api/docs" -ForegroundColor White
Write-Host "  • Frontend:     http://localhost:5173" -ForegroundColor White
Write-Host "  • QuestDB:      http://localhost:9000" -ForegroundColor White
Write-Host ""
Write-Host "⌨️  Press any key to open Frontend in browser..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Open frontend in default browser
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "✅ Done! System is running." -ForegroundColor Green
Write-Host ""


