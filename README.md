# Synthetic EV Scheduling Study Reproduction

This repository contains a standalone Python script that synthesizes data and
visualizations inspired by the EV scheduling paper referenced in `1.pdf`. The
script focuses on two key contributions from the article:

1. **Peak-valley reduction and operating cost minimization** through an improved
   grey wolf optimization (IGEO) algorithm that coordinates EV charging and
   discharging.
2. **EV battery life protection** via K-means clustering that assigns vehicles to
   power groups based on battery statistics.

Running the script produces **11 visuals** (8 analytical plots plus 3 auxiliary
figures) and **3 tables** that resemble the trends reported in the paper while
remaining reproducible and synthetic.

## Generated assets

`python src/generate_ev_study.py` creates the following structure:

```
outputs/
├── figures/
│   ├── fig1_load_profile.png
│   ├── …
│   └── fig11_soc_histogram.png
└── tables/
    ├── table1_time_of_use.csv / .md
    ├── table2_peak_valley.csv / .md
    └── table3_life_loss.csv / .md
```

Each figure mirrors a specific illustration from the reference article, such as
regulation signals, algorithm convergence, clustered fleet distributions, and
battery degradation trends.

## Usage

1. Install the required dependencies (listed in `requirements.txt`).
2. Execute `python src/generate_ev_study.py`.
3. Review the generated figures and tables in the `outputs/` directory.

> **Note:** The random seed is fixed to ensure reproducible outputs.

## Requirements

The project depends on standard scientific Python packages. You can install
them with:

```bash
pip install -r requirements.txt
```

The script only requires a basic Python 3.10+ environment with NumPy, Pandas,
and Matplotlib available.
