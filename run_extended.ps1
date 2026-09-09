# Wrapper para la corrida MCMC extendida (evita el deadlock de handles
# redirigidos de Start-Process -NoNewWindow). Log en EXT_MCMC.log.
$ErrorActionPreference = "Continue"
Set-Location "d:\DVCH13-1"
$env:PYTHONIOENCODING = "utf-8"
& "d:\DVCH13-1\.venv\Scripts\python.exe" dvch_mcmc_extended_convergence.py *>> "d:\DVCH13-1\EXT_MCMC.log"
Add-Content -Path "d:\DVCH13-1\EXT_MCMC.log" -Value ("EXIT_CODE=" + $LASTEXITCODE)
