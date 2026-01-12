@echo off
title Savage AI Controller
color 0C

echo.
echo    _____ ___ __   __  ___   ____  ______      _       ____ 
echo   / ___//   ^|\ \ / / /   ^| / ___^|/ ____/     / \     ^|_  _^|
echo   \__ \/ /^| ^| \ V / / /^| ^|/ / _ /  _/       / _ \     _^| ^|_ 
echo  ___/ / ___ ^|  \ / / ___ ^| ^|_^| ^|/ /___     / ___ ^\   ^|_____^|
echo /____/_/  ^|_^|  \/ /_/  ^|_^|\____/_____/    /_/   \_\  
echo.
echo ==========================================================
echo [INFO] Initializing Savage Protocol...
echo [INFO] Connecting to Local AI...
echo [INFO] Starting Dashboard at http://localhost:5000
echo.
echo    Created by Logicalgamer
echo    Coded by Gemini ^& Google Antigravity
echo ==========================================================
echo.

:start
python execution/discord_bot.py
echo.
echo [Info] Bot stopped. Restarting in 3 seconds...
timeout /t 3
goto start
