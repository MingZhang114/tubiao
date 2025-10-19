"""Synthetic reproduction of EV optimization study figures and tables.

This module creates synthetic data that mimics the trends reported in the
referenced EV scheduling paper. It generates 8 figures and 3 tables that
approximate the visual appearance of the publication while remaining entirely
self-contained and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")


SEED = 7
RNG = np.random.default_rng(SEED)
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


@dataclass
class OptimizationHistory:
    """Container for meta-heuristic optimization histories."""

    iterations: np.ndarray
    fitness: Dict[str, np.ndarray]


@dataclass
class LoadProfiles:
    """Container for pre- and post-optimization load profiles."""

    timeline: pd.Index
    baseline: np.ndarray
    optimized: np.ndarray


@dataclass
class RegulationSignal:
    """Charging and discharging instructions for the EV fleet."""

    timeline: pd.Index
    power: np.ndarray


@dataclass
class BatteryDegradation:
    """Battery degradation trajectories for different strategies."""

    cycles: np.ndarray
    degradation: Dict[str, np.ndarray]


@dataclass
class TrackingMetrics:
    """Time series of tracking deviations for evaluation."""

    timeline: pd.Index
    deviation: Dict[str, np.ndarray]


@dataclass
class ClusterResult:
    """Result of the synthetic K-means grouping."""

    points: np.ndarray
    labels: np.ndarray
    centers: np.ndarray


def _time_index(periods: int = 96) -> pd.Index:
    """Return a 15-minute resolution timeline for a single day."""

    return pd.date_range("2024-01-01", periods=periods, freq="15min")


def create_load_profiles() -> LoadProfiles:
    """Create synthetic load profiles before and after IGEO optimization."""

    timeline = _time_index()
    hours = np.linspace(0, 24, len(timeline))
    base_shape = 420 + 120 * np.sin((hours - 7) * np.pi / 12)
    evening_peak = 160 * np.exp(-0.5 * ((hours - 20) / 1.7) ** 2)
    morning_peak = 90 * np.exp(-0.5 * ((hours - 8.2) / 2.1) ** 2)
    random_variation = RNG.normal(0, 12, size=len(hours))
    baseline = base_shape + evening_peak + morning_peak + random_variation

    # IGEO optimization flattens peaks and elevates valleys with mild noise.
    smoothing = np.convolve(baseline, np.ones(5) / 5, mode="same")
    valley_boost = 40 * np.exp(-0.5 * ((hours - 3.5) / 2.5) ** 2)
    optimized = 0.94 * smoothing + valley_boost

    return LoadProfiles(timeline=timeline, baseline=baseline, optimized=optimized)


def create_regulation_signal(load_profiles: LoadProfiles) -> RegulationSignal:
    """Create a synthetic charge/discharge signal resembling Fig. 3."""

    hours = np.linspace(0, 24, len(load_profiles.timeline))
    base_signal = 120 * np.sin((hours - 12) * np.pi / 6)
    corrective = 80 * np.sin((hours - 19) * np.pi / 3)
    noise = RNG.normal(0, 18, size=len(hours))
    power = base_signal + corrective + noise
    power = np.clip(power, -260, 260)

    return RegulationSignal(timeline=load_profiles.timeline, power=power)


def create_optimization_history() -> OptimizationHistory:
    """Generate monotonically decreasing fitness curves for algorithms."""

    iterations = np.arange(1, 101)
    fitness = {}
    baseline_curve = 0.38 * np.exp(-iterations / 47)
    for label, lag, variability in [
        ("GA", 0.22, 0.015),
        ("PSO", 0.18, 0.012),
        ("WOA", 0.16, 0.013),
        ("GEO", 0.11, 0.01),
        ("IGEO", 0.08, 0.007),
    ]:
        curve = baseline_curve + lag
        noise = RNG.normal(0, variability, size=len(iterations))
        fitness[label] = np.maximum.accumulate((curve + noise)[::-1])[::-1]
    return OptimizationHistory(iterations=iterations, fitness=fitness)


def create_battery_degradation() -> BatteryDegradation:
    """Create synthetic battery degradation trajectories for strategies."""

    cycles = np.arange(0, 550, 10)
    degradation = {}
    base = 2.0 + 0.0009 * cycles + 0.000002 * cycles**2
    strategies = {
        "Uncontrolled": 1.0,
        "Time-of-Use": 0.86,
        "Peak Shaving": 0.78,
        "Proposed": 0.65,
    }
    for label, multiplier in strategies.items():
        shape = base * multiplier
        ripple = RNG.normal(0, 0.035 * multiplier, size=len(cycles))
        degradation[label] = np.clip(shape + ripple, 0, None)
    return BatteryDegradation(cycles=cycles, degradation=degradation)


def create_tracking_metrics(timeline: pd.Index) -> TrackingMetrics:
    """Create standard deviation tracking results for two signals."""

    hours = np.linspace(0, 24, len(timeline))
    base = 8 + 6 * np.sin((hours - 7) * np.pi / 8)
    geo = base - 2.2 * np.exp(-0.5 * ((hours - 9) / 2.4) ** 2)
    igeo = geo - 1.1 * np.exp(-0.5 * ((hours - 19) / 2.0) ** 2)
    jitter = RNG.normal(0, 0.4, size=len(hours))
    return TrackingMetrics(
        timeline=timeline,
        deviation={
            "Original": np.abs(base + 0.3 * jitter),
            "After GEO": np.abs(geo + 0.25 * jitter),
            "After IGEO": np.abs(igeo + 0.2 * jitter),
        },
    )


def synthetic_kmeans(points: np.ndarray, k: int, iterations: int = 30) -> ClusterResult:
    """Simple implementation of K-means clustering for reproducibility."""

    centers = points[RNG.choice(len(points), size=k, replace=False)]
    for _ in range(iterations):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        for idx in range(k):
            members = points[labels == idx]
            if len(members) > 0:
                centers[idx] = members.mean(axis=0)
    return ClusterResult(points=points, labels=labels, centers=centers)


def create_ev_fleet_groups(samples: int = 120, k: int = 4) -> ClusterResult:
    """Generate synthetic EV fleet statistics and cluster them."""

    capacity = RNG.normal(58, 6, size=samples)
    soc = np.clip(RNG.normal(0.45, 0.15, size=samples), 0.1, 0.9)
    mileage = np.clip(RNG.normal(42, 18, size=samples), 5, None)
    points = np.column_stack((capacity, soc, mileage))
    return synthetic_kmeans(points, k=k)


def create_power_distribution(labels: np.ndarray, timeline: pd.Index) -> pd.DataFrame:
    """Create clustered power allocation schedule for the EV fleet."""

    hours = np.linspace(0, 24, len(timeline))
    base_shape = 35 + 12 * np.sin((hours - 8) * np.pi / 10)
    cluster_modifiers = {
        0: 1.3,
        1: 1.1,
        2: 0.9,
        3: 0.7,
    }
    power = {
        f"Group {cluster + 1}":
        np.clip(base_shape * cluster_modifiers.get(cluster, 1.0) + RNG.normal(0, 1.5, len(hours)), 5, 110)
        for cluster in np.unique(labels)
    }
    return pd.DataFrame(power, index=timeline)


def save_table(data: pd.DataFrame, filename: str) -> None:
    """Save a table to CSV and Markdown for easy inspection."""

    csv_path = TABLE_DIR / f"{filename}.csv"
    md_path = TABLE_DIR / f"{filename}.md"
    data.to_csv(csv_path, index=False)
    data.to_markdown(md_path, index=False)


def plot_load_profiles(load_profiles: LoadProfiles) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(load_profiles.timeline, load_profiles.baseline, label="Original load", linewidth=1.6)
    ax.plot(load_profiles.timeline, load_profiles.optimized, label="After IGEO", linewidth=1.6)
    ax.set_ylabel("Power / kW")
    ax.set_xlabel("Time")
    ax.set_title("Comparison of daily load curves")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(FIGURE_DIR / "fig1_load_profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_regulation_signal(signal: RegulationSignal) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(signal.timeline, signal.power, color="#1f77b4", linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Power / kW")
    ax.set_xlabel("Time")
    ax.set_title("EV regulation signal under IGEO optimization")
    fig.autofmt_xdate()
    fig.savefig(FIGURE_DIR / "fig2_regulation_signal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fitness_history(history: OptimizationHistory) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, values in history.fitness.items():
        ax.plot(history.iterations, values, label=label, linewidth=1.3)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fitness function value")
    ax.set_title("Fitness convergence of different algorithms")
    ax.set_xlim(1, history.iterations[-1])
    ax.legend()
    fig.savefig(FIGURE_DIR / "fig3_fitness_history.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_performance_radar(metrics: pd.DataFrame) -> None:
    categories = metrics.columns[1:]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    for _, row in metrics.iterrows():
        values = row[categories].to_numpy()
        values = np.concatenate([values, values[:1]])
        ax.plot(angles, values, label=row["Algorithm"], linewidth=1.2)
        ax.fill(angles, values, alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title("Normalized performance comparison")
    ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.1))
    fig.savefig(FIGURE_DIR / "fig4_performance_radar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_degradation_curves(degradation: BatteryDegradation) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, curve in degradation.degradation.items():
        ax.plot(degradation.cycles, curve, label=label, linewidth=1.5)
    ax.set_xlabel("Charge-discharge cycles")
    ax.set_ylabel("Capacity loss / %")
    ax.set_title("EV battery degradation under different strategies")
    ax.legend()
    fig.savefig(FIGURE_DIR / "fig5_battery_degradation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_charge_discharge(reg_signal: RegulationSignal) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    pre_signal = np.clip(reg_signal.power + RNG.normal(0, 35, len(reg_signal.power)) - 60, -260, 260)
    ax.plot(reg_signal.timeline, pre_signal, label="Original control", linewidth=1.1)
    ax.plot(reg_signal.timeline, reg_signal.power, label="After IGEO", linewidth=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Power / kW")
    ax.set_title("EV charge and discharge schedules")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(FIGURE_DIR / "fig6_charge_discharge.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tracking_metrics(metrics: TrackingMetrics) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, series in metrics.deviation.items():
        ax.plot(metrics.timeline, series, label=label, linewidth=1.2)
    ax.set_ylabel("Standard deviation / kW")
    ax.set_xlabel("Time")
    ax.set_title("Tracking standard deviation of different strategies")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(FIGURE_DIR / "fig7_tracking_deviation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_scatter(cluster: ClusterResult) -> None:
    fig = plt.figure(figsize=(7, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.Set2(cluster.labels / (cluster.centers.shape[0] - 0.5))
    ax.scatter(cluster.points[:, 0], cluster.points[:, 1], cluster.points[:, 2], c=colors, s=26, alpha=0.85)
    ax.scatter(
        cluster.centers[:, 0],
        cluster.centers[:, 1],
        cluster.centers[:, 2],
        c="black",
        s=120,
        marker="*",
        label="Cluster centers",
    )
    ax.set_xlabel("Battery capacity / kWh")
    ax.set_ylabel("Initial SOC")
    ax.set_zlabel("Expected daily mileage / km")
    ax.set_title("K-means grouping of EV fleet")
    ax.legend(loc="upper left")
    fig.savefig(FIGURE_DIR / "fig8_kmeans_groups.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_tables(load_profiles: LoadProfiles, history: OptimizationHistory, degradation: BatteryDegradation) -> None:
    """Create CSV and Markdown versions of the three tables."""

    # Table 1: Time-of-use prices.
    table1 = pd.DataFrame(
        {
            "Time Period": [
                "00:00-06:00",
                "06:00-10:00",
                "10:00-16:00",
                "16:00-18:00",
                "18:00-22:00",
                "22:00-24:00",
            ],
            "Price (yuan/kWh)": [0.38, 0.62, 0.54, 0.71, 0.92, 0.48],
        }
    )
    save_table(table1, "table1_time_of_use")

    # Table 2: Peak-valley disparity and operating cost comparison.
    disparity = load_profiles.baseline.max() - load_profiles.baseline.min()
    table2 = pd.DataFrame(
        {
            "Algorithm": ["GA", "PSO", "WOA", "GEO", "IGEO"],
            "Peak-valley disparity (kW)": np.round(
                disparity * np.array([0.91, 0.86, 0.82, 0.74, 0.61]), 2
            ),
            "Disparity rate (%)": [9.2, 13.4, 17.8, 25.6, 38.9],
            "Operational cost (yuan/h)": [548, 533, 527, 508, 487],
        }
    )
    save_table(table2, "table2_peak_valley")

    # Table 3: Battery life loss under different strategies.
    final_loss = {label: values[-1] for label, values in degradation.degradation.items()}
    table3 = pd.DataFrame(
        {
            "Strategy": list(final_loss.keys()),
            "Energy delivered (MWh)": [8.2, 7.9, 7.6, 7.4],
            "Life loss (%)": np.round(list(final_loss.values()), 2),
            "Improvement vs. uncontrolled (%)": [0, 12.4, 19.6, 31.3],
        }
    )
    save_table(table3, "table3_life_loss")


def generate_performance_metrics(history: OptimizationHistory) -> pd.DataFrame:
    """Create normalized performance metrics for the radar chart."""

    algorithms = list(history.fitness.keys())

    reduction_steps = []
    for values in history.fitness.values():
        below_threshold = np.nonzero(values < values[0] * 0.22)[0]
        reduction_steps.append(below_threshold[0] + 1 if len(below_threshold) else len(values))
    convergence_speed = [1 / step for step in reduction_steps]
    best_fitness = [values.min() for values in history.fitness.values()]
    peak_shaving = np.linspace(0.62, 0.9, len(algorithms))[::-1]

    df = pd.DataFrame(
        {
            "Algorithm": algorithms,
            "Convergence": np.interp(
                convergence_speed,
                (min(convergence_speed), max(convergence_speed)),
                (0.4, 0.95),
            ),
            "Fitness": np.interp(
                best_fitness,
                (min(best_fitness), max(best_fitness)),
                (0.9, 0.45),
            ),
            "Peak shaving": peak_shaving,
            "Cost saving": np.linspace(0.55, 0.95, len(algorithms)),
        }
    )
    return df





def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(exist_ok=True)

    load_profiles = create_load_profiles()
    reg_signal = create_regulation_signal(load_profiles)
    history = create_optimization_history()
    degradation = create_battery_degradation()
    tracking = create_tracking_metrics(load_profiles.timeline)
    cluster = create_ev_fleet_groups()
    power_distribution = create_power_distribution(cluster.labels, load_profiles.timeline)

    generate_tables(load_profiles, history, degradation)

    plot_load_profiles(load_profiles)
    plot_regulation_signal(reg_signal)
    plot_fitness_history(history)
    metrics = generate_performance_metrics(history)
    plot_performance_radar(metrics)
    plot_degradation_curves(degradation)
    plot_charge_discharge(reg_signal)
    plot_tracking_metrics(tracking)
    plot_cluster_scatter(cluster)

    # Additional figure showing clustered power dispatch profile.
    fig, ax = plt.subplots(figsize=(7, 4))
    power_distribution.plot(ax=ax, linewidth=1.2)
    ax.set_ylabel("Power / kW")
    ax.set_xlabel("Time")
    ax.set_title("Clustered EV power dispatch schedule")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.autofmt_xdate()
    fig.savefig(FIGURE_DIR / "fig9_power_dispatch.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Final figure: overlay of original vs optimized load to highlight reduction.
    fig, ax = plt.subplots(figsize=(7, 4))
    delta = load_profiles.baseline - load_profiles.optimized
    ax.fill_between(load_profiles.timeline, 0, delta, where=delta > 0, color="#ff7f0e", alpha=0.4, label="Peak reduction")
    ax.fill_between(load_profiles.timeline, 0, delta, where=delta <= 0, color="#2ca02c", alpha=0.4, label="Valley filling")
    ax.plot(load_profiles.timeline, delta, color="#1f77b4", linewidth=1.2)
    ax.set_ylabel("Delta power / kW")
    ax.set_xlabel("Time")
    ax.set_title("Effect of IGEO scheduling on net load")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(FIGURE_DIR / "fig10_load_delta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 11: histogram of SOC distribution for clustered groups.
    fig, ax = plt.subplots(figsize=(7, 4))
    soc_values = cluster.points[:, 1]
    for label in np.unique(cluster.labels):
        ax.hist(
            soc_values[cluster.labels == label],
            bins=np.linspace(0, 1, 11),
            alpha=0.6,
            label=f"Group {label + 1}",
        )
    ax.set_xlabel("Initial SOC")
    ax.set_ylabel("Number of EVs")
    ax.set_title("SOC distribution of clustered EV groups")
    ax.legend()
    fig.savefig(FIGURE_DIR / "fig11_soc_histogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
