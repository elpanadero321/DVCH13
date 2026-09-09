# ============================================================
# DVCH — Batería de pruebas completa (física completa + MCMC)
# Ejecuta todo el stack local de forma secuencial y deja un log
# maestro con código de salida y duración de cada etapa.
# Uso:  powershell -NoProfile -ExecutionPolicy Bypass -File run_full_battery.ps1
# ============================================================
$ErrorActionPreference = "Continue"
Set-Location "d:\DVCH13-1"
$py  = "d:\DVCH13-1\.venv\Scripts\python.exe"
$log = "d:\DVCH13-1\BATTERY_MASTER.log"
Set-Content -Path $log -Value "=== DVCH FULL BATTERY START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

function Run-Stage([string]$name, [string[]]$argv) {
    $t0 = Get-Date
    Add-Content -Path $log -Value ""
    Add-Content -Path $log -Value ">>> STAGE: $name  @ $(Get-Date -Format 'HH:mm:ss')"
    & $py @argv 2>&1 | ForEach-Object { Add-Content -Path $log -Value ("    " + $_) }
    $code = $LASTEXITCODE
    $dt = (Get-Date) - $t0
    Add-Content -Path $log -Value ("<<< STAGE DONE: $name  exit=$code  duration=$([int]$dt.TotalSeconds)s")
}

# --- A. Unit tests (17) ---
Run-Stage "A_unittest_17" @("-m", "unittest", "tests.test_dvch", "-v")

# --- C. Validaciones de física (rápidas) ---
Run-Stage "C1_double_slit"            @("dvch_double_slit.py")
Run-Stage "C2_relativistic_perturb"   @("dvch_relativistic_perturbation_validation.py")
Run-Stage "C3_cmb_readiness"          @("dvch_cmb_class_camb_mcmc.py")
Run-Stage "C4_planck_preflight"       @("dvch_planck_preflight.py")
Run-Stage "C5_camb_background"        @("dvch_camb_background.py")
Run-Stage "C6_boltzmann_backend"      @("dvch_boltzmann_backend.py")
Run-Stage "C7_modified_solver"        @("dvch_modified_boltzmann_solver.py")
Run-Stage "C8_growth_sigma8"         @("dvch_growth_diagnostic.py")
Run-Stage "C9_robustness_scan"        @("dvch_robustness_scan.py")
Run-Stage "C10_realdata_diagnostics"  @("dvch_realdata_diagnostics.py")
Run-Stage "C11_joint_realdata_fit"    @("dvch_joint_realdata_fit.py")
Run-Stage "C12_colab_demo"            @("dvch_colab_simple.py")

# --- B. MCMC (los más largos al final) ---
Run-Stage "B1_mcmc_convergence"       @("dvch_mcmc_convergence.py")
Run-Stage "B2_full_mcmc_pipeline"     @("dvch_full_mcmc_pipeline.py")

Add-Content -Path $log -Value ""
Add-Content -Path $log -Value "=== DVCH FULL BATTERY END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
