# Blocks accidental commits of large/secret/raw-data files.
$dangerous = @(
    "data/raw/lalonde_nsw.csv",
    "data/raw/lalonde_cps.csv",
    "data/raw/*.dta",
    "models/*.joblib",
    "models/*.pkl",
    ".env",
    "mlruns",
    "CLAUDE.md",
    "MANUAL_TASKS.md"
)
$staged = git diff --cached --name-only
foreach ($f in $staged) {
    foreach ($pat in $dangerous) {
        if ($f -like $pat) {
            Write-Host "Refusing to commit dangerous path: $f"
            exit 1
        }
    }
}
exit 0
