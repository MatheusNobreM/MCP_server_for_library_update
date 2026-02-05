Remove-Item ".\data\demo.db" -ErrorAction SilentlyContinue
Remove-Item ".\memory.db" -ErrorAction SilentlyContinue
Write-Host "Reset concluído (demo.db e memory.db removidos)."
