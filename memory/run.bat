@echo off
chcp 65001 >nul
echo Starting Local Memory MCP Server...
cd /d "%~dp0"
python server.py
pause
