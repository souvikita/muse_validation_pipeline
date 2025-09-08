# MUSE Validation Pipeline

Minimal toolkit to (a) fetch / prepare numerical solar atmosphere model outputs and (b) compare them with observations from:
- SDO / AIA
- SDO / HMI
- Hinode / EIS
- IRIS

Goal: provide consistent, lightweight validation steps so numerical models can be assessed and refined for forthcoming MUSE data modeling workflows.

Internal coordination: linked to the `muse_science` Slack workspace (private channel: `#museval`).

## Installation
_Better to install under a virtual conda/mamba environment_ e.g.: 
`conda create -n your_env_name`

Clone and install in editable mode:

```bash
git clone https://github.com/souvikita/muse_validation_pipeline.git
cd muse_validation_pipeline
pip install -e .
```

## Quick Concept

1. Retrieve / point to a simulation in the form of a Velocity DEM (e.g. Bifrost, MHD model, etc.).
2. Synthesize observables by taking into account the different instrumental response functions. 
3. Select corresponding observation sets (AIA / HMI / EIS / IRIS). There are functions to output observations in the form of .txt files.
4. Run comparison / diagnostic routines (intensity synthesis, line ratios, basic statistics, etc.).
5. Produce a small summary plots.

(Actual commands will be documented once interfaces stabilize.)

## Usage (Placeholder)

Usage examples will be added once the first comparison scripts are finalized.  
For now, see `working_codes` inside the branch `bose`.


## Contributing

Small, focused improvements welcome.  
Open an issue or draft a PR once there is a clearer module layout.


---

Questions / ideas: open an issue.

### Thanks to ChatGPT 5.0 for helping to create this nice little ReadMe.
