# Synthetic EV Scheduling Study Reproduction

This repository contains a standalone Python script that synthesizes data and
visualizations inspired by the EV scheduling paper referenced in `1.pdf`. The
script focuses on two key contributions from the article:

1. **Peak-valley reduction and operating cost minimization** through an improved
   grey wolf optimization (IGEO) algorithm that coordinates EV charging and
   discharging.
2. **EV battery life protection** via K-means clustering that assigns vehicles to
   power groups based on battery statistics.

Running the script produces **8 figures** and **3 tables** that resemble the
trends reported in the paper while remaining reproducible and synthetic.

## Generated assets

`python src/run_ev_iLSHADE_plus_v4_fixed.py` creates the following structure:

```
outputs/
├── fig_02_ev_count.png
├── fig_03_ev_signal_iLSHADEplus.png
├── fig_04_convergence.png
├── fig_05_soc_group_D_before_after.png
├── fig_06_signal_vs_response.png
├── fig_07_tracking_deviation.png
├── fig_08_load_before_after_day1.png
├── fig_09_load_before_after_3days.png
├── tab_02_tou_prices.csv
├── tab_03_metrics.csv
└── tab_05_life_loss_summary.csv
```

Each figure mirrors a specific illustration from the reference article, such as
regulation signals, algorithm convergence, clustered fleet distributions, and
battery degradation trends. The script automatically creates the `outputs/`
directory if it does not already exist.

## Usage

1. Install the required dependencies (listed in `requirements.txt`).
2. Execute `python src/run_ev_iLSHADE_plus_v4_fixed.py`.
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
