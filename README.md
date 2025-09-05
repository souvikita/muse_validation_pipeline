# MUSE Validation Pipeline

Minimal toolkit to (a) fetch / prepare numerical solar atmosphere model outputs and (b) compare them with observations from:
- SDO / AIA
- SDO / HMI
- Hinode / EIS
- IRIS

Goal: provide consistent, lightweight validation steps so numerical models can be assessed and refined for forthcoming MUSE data modeling workflows.

Internal coordination: linked to the `muse_science` Slack workspace (private channel: `#museval`).

## Installation

Clone and install in editable mode:

```bash
git clone https://github.com/souvikita/muse_validation_pipeline.git
cd muse_validation_pipeline
pip install -e .
```

(If you later add optional dependencies, note them here.)

## Quick Concept

1. Retrieve / point to a simulation (e.g. Bifrost, MHD model, etc.).
2. Select corresponding observation sets (AIA / HMI / EIS / IRIS).
3. Run comparison / diagnostic routines (intensity synthesis, line ratios, basic statistics, etc.).
4. Produce a small summary (numbers / plots).

(Actual commands will be documented once interfaces stabilize.)

## Usage (Placeholder)

Usage examples will be added once the first comparison scripts are finalized.  
For now, see individual modules / scripts as they are added.


## Contributing

Small, focused improvements welcome.  
Open an issue or draft a PR once there is a clearer module layout.



---

Questions / ideas: open an issue.

### Thanks to ChatGPT 5.0 for helping to create this nice little ReadMe.
