$PackagesPath = "C:\cosmo\cobaya_packages"
$ClassyPath = "C:\cosmo\class_public"
$OutputRoot = "C:\Users\danie\.copilot\repos\copilot-worktrees\dvch13\elpanadero321-congenial-fishstick\dvch_full_pipeline\chains\dvch_full"

python .\dvch_full_pipeline\prepare_bundle.py --packages-path $PackagesPath --classy-path $ClassyPath --output-root $OutputRoot
python .\dvch_full_pipeline\run_full_pipeline.py --packages-path $PackagesPath --classy-path $ClassyPath --output-root $OutputRoot
