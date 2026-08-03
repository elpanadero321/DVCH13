DVCH full production scaffold
============================

Purpose
- Keep the heavy Planck + Pantheon+ + BAO Cobaya/classy run separated from the lighter root-level diagnostics.
- Leave the production bundle ready to execute later without starting the expensive chain now.

Main files
- prepare_bundle.py: writes the bundle-local YAML, runtime manifest, readiness table, and example launcher.
- run_full_pipeline.py: executes the heavy Cobaya run later, once external dependencies are installed.
- dvch_full_cobaya.yaml: full Planck 2018 + Pantheon+ + BAO Cobaya configuration.
- dvch_full_runtime_manifest.json: exact runtime checklist, paths, and bridge metadata.
- status\dvch_full_pipeline_status.csv / .png: local readiness outputs for this folder only.
- chains\: default output directory for the future heavy chain.

External pieces still required before the real run
- Patched CLASS/classy backend with the DVCH bridge keys dvch_n, dvch_beta, dvch_model, dvch_use_exact_q.
- Cobaya packages path containing Planck 2018 clik, Pantheon+, and BAO likelihood data.

Default workflow
1. Edit run_full_pipeline_example.ps1 or pass the paths directly on the command line.
2. Run: python .\dvch_full_pipeline\prepare_bundle.py --packages-path <...> --classy-path <...>
3. Run: python .\dvch_full_pipeline\run_full_pipeline.py --packages-path <...> --classy-path <...>
4. Confirm the final chains satisfy R-1 < 0.02.

Current state
- This folder is prepared but not executed.
- The already completed lightweight and intermediate validations remain at repository root.
