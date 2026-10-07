# Python_Clean_Prepare_Data_Course/Python_Clean_Prepare_Data_Course.py

"""Self-paced course on cleaning and preparing scientific data with Python.

Run this file with the project virtual environment:

    /Users/victorbui/venvs/ai312/bin/python Python_Clean_Prepare_Data_Course.py

The script creates a reproducible light-curve dataset, applies each cleaning
stage, validates the results, and writes tables and figures to ``output``.
"""

# Import the libraries used throughout the course.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Resolve paths from this file so execution does not depend on the shell folder.
COURSE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = COURSE_DIR / "output"


def section(title):
    """Print a visible lesson heading.

    ``title`` is a short section name. The function changes console output only
    and has no effect on the data or files.
    """
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def create_light_curve(seed=42):
    """Create a reproducible signal with known quality problems.

    The returned DataFrame contains time in days and observed brightness.
    It includes slow drift, two transit-like dips, Gaussian noise, isolated
    spikes, one short gap, one long gap, and a sentinel missing-value code.
    """
    # Use uneven sampling so interpolation and time-aware windows matter.
    generator = np.random.default_rng(seed)
    time_steps = generator.uniform(0.007, 0.013, 520)
    time_days = np.cumsum(time_steps)
    time_days = 5 * (time_days - time_days.min()) / np.ptp(time_days)

    # Combine a slow baseline, two dips, and measurement noise.
    baseline = 1 + 0.0025 * time_days - 0.0007 * time_days**2
    first_transit = -0.012 * np.exp(-0.5 * ((time_days - 2.00) / 0.075) ** 2)
    second_transit = -0.009 * np.exp(-0.5 * ((time_days - 4.15) / 0.060) ** 2)
    noise = generator.normal(0, 0.0012, time_days.size)
    brightness = baseline + first_transit + second_transit + noise

    # Add known spikes for evaluating the outlier detector.
    injected_outliers = np.array([55, 173, 310, 462])
    brightness[injected_outliers] += np.array([0.025, -0.022, 0.028, -0.024])

    # Add small and large gaps plus a numeric sentinel used by some instruments.
    brightness[120:123] = np.nan
    brightness[355:377] = np.nan
    brightness[250] = -999

    return pd.DataFrame({"time_days": time_days, "brightness": brightness})


def standardize_missing(data, column, sentinels):
    """Return a copy with known sentinel codes replaced by ``NaN``.

    ``data`` is a DataFrame, ``column`` identifies the measured variable, and
    ``sentinels`` lists numeric codes that mean missing. The input is unchanged.
    """
    standardized = data.copy()
    standardized[column] = standardized[column].replace(sentinels, np.nan)
    return standardized


def rolling_mad_outliers(values, window=41, threshold=5):
    """Detect local outliers with a rolling median and median deviation.

    ``values`` is a numeric Series. ``window`` is an odd count of nearby
    samples and ``threshold`` controls sensitivity. The returned Boolean Series
    is aligned to the input. Missing values never count as detected outliers.
    """
    # A rolling center prevents a slow trend from being mistaken for an outlier.
    center = values.rolling(window, center=True, min_periods=9).median()
    absolute_deviation = (values - center).abs()
    local_mad = absolute_deviation.rolling(
        window, center=True, min_periods=9
    ).median()

    # Convert MAD to a robust standard-deviation estimate for near-normal noise.
    robust_scale = 1.4826 * local_mad
    fallback_scale = 1.4826 * np.nanmedian(absolute_deviation)
    robust_scale = robust_scale.mask(robust_scale <= 0, fallback_scale)
    return ((absolute_deviation > threshold * robust_scale) & values.notna()).fillna(
        False
    )


def interpolate_small_gaps(data, column, limit=5):
    """Fill only short internal gaps by linear interpolation over time.

    ``limit`` is the maximum number of consecutive missing samples to fill.
    Long gaps and missing values at either boundary remain missing so the result
    does not imply unsupported observations.
    """
    interpolated = data.copy()
    timed = interpolated.set_index("time_days")[column]

    # Identify complete missing runs so a long gap is not partly filled at its edges.
    missing = timed.isna()
    run_id = missing.ne(missing.shift()).cumsum()
    run_size = missing.groupby(run_id).transform("sum")
    fillable = missing & (run_size <= limit)

    # Calculate candidate values over real time, then copy only supported runs.
    candidate = timed.interpolate(method="index", limit_area="inside")
    timed.loc[fillable] = candidate.loc[fillable]
    interpolated[column] = timed.to_numpy()
    return interpolated


def add_normalizations(data, column):
    """Return a copy with centered, z-score, relative, and range-scaled columns.

    Statistics ignore missing values. Constant inputs raise an error because
    their standard deviation and range cannot support these transformations.
    """
    normalized = data.copy()
    values = normalized[column]
    mean_value = values.mean()
    standard_deviation = values.std(ddof=0)
    value_range = values.max() - values.min()
    if standard_deviation == 0 or value_range == 0:
        raise ValueError("Normalization requires nonconstant data.")

    # Each form answers a different analytical question.
    normalized["centered"] = values - mean_value
    normalized["z_score"] = (values - mean_value) / standard_deviation
    normalized["relative_brightness"] = (values - mean_value) / mean_value
    normalized["scaled_minus1_to1"] = 2 * (values - values.min()) / value_range - 1
    return normalized


def polynomial_detrend(data, column, degree=3, mask=None):
    """Fit and subtract a polynomial trend from a measured signal.

    ``mask`` can exclude scientifically important events from the trend fit.
    The returned DataFrame adds ``trend`` and ``detrended`` columns while
    preserving missing samples and the original measurement.
    """
    detrended = data.copy()
    valid = detrended[column].notna().to_numpy()
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    # Center and scale time before fitting to improve polynomial conditioning.
    time = detrended["time_days"].to_numpy()
    time_center = np.nanmean(time[valid])
    time_scale = np.nanstd(time[valid])
    scaled_time = (time - time_center) / time_scale
    coefficients = np.polyfit(
        scaled_time[valid], detrended.loc[valid, column], degree
    )
    trend = np.polyval(coefficients, scaled_time)
    detrended["trend"] = trend
    detrended["detrended"] = detrended[column] - trend
    return detrended


def time_window_smooth(data, column, before_days, after_days, method="mean"):
    """Smooth a signal using an asymmetric window measured in days.

    The window for each row spans ``before_days`` backward and ``after_days``
    forward. ``method`` is ``mean`` or ``median``. Missing values are ignored,
    and the returned Series retains the original row index.
    """
    if method not in {"mean", "median"}:
        raise ValueError("method must be 'mean' or 'median'")

    # Use actual time distances so irregular sample spacing is handled correctly.
    times = data["time_days"].to_numpy()
    values = data[column].to_numpy()
    result = np.full(values.size, np.nan)
    reducer = np.nanmean if method == "mean" else np.nanmedian
    for position, current_time in enumerate(times):
        in_window = (
            (times >= current_time - before_days)
            & (times <= current_time + after_days)
        )
        window_values = values[in_window]
        if np.isfinite(window_values).any():
            result[position] = reducer(window_values)
    return pd.Series(result, index=data.index, name=f"smoothed_{method}")


def plot_raw_and_outliers(data, outlier_mask, destination):
    """Save the raw signal and marked outliers to ``destination``.

    Time uses days and brightness uses relative units. The function writes one
    PNG file and closes the figure so repeated runs do not retain plot state.
    """
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(data["time_days"], data["brightness"], lw=1, label="Raw data")
    axis.scatter(
        data.loc[outlier_mask, "time_days"],
        data.loc[outlier_mask, "brightness"],
        color="crimson",
        marker="x",
        s=45,
        label="Detected outlier",
        zorder=3,
    )
    axis.set(
        xlabel="Time (days)",
        ylabel="Relative brightness",
        title="Local outlier detection",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def plot_processing_stages(data, destination):
    """Save a four-panel view of raw, normalized, detrended, and smoothed data.

    ``data`` must contain the named processing columns produced by this course.
    The plot provides a visual quality check; it does not replace assertions.
    """
    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    stages = [
        ("brightness_clean", "Cleaned brightness", "Relative brightness"),
        ("relative_brightness", "Centered and scaled", "Fractional change"),
        ("detrended", "Cubic trend removed", "Residual"),
        ("smoothed", "Centered 0.10-day mean", "Smoothed residual"),
    ]
    for axis, (column, title, ylabel) in zip(axes, stages):
        axis.plot(data["time_days"], data[column], lw=1)
        axis.set(title=title, ylabel=ylabel)
    axes[-1].set_xlabel("Time (days)")
    figure.tight_layout()
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def run_course():
    """Execute every lesson and save the final dataset and figures.

    The function prints intermediate checks, writes only inside ``output``, and
    returns the final DataFrame so learners can continue exploring in Python.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Lesson 1: inspect raw data before choosing any cleaning method.
    section("1. Inspect raw data")
    raw = create_light_curve()
    print(raw.head().to_string(index=False))
    print("Rows:", len(raw))
    print("Explicit missing values:", raw["brightness"].isna().sum())
    print("Minimum before standardization:", raw["brightness"].min())

    # Lesson 2: standardize missing-value codes before calculating statistics.
    section("2. Locate and standardize missing values")
    standardized = standardize_missing(raw, "brightness", sentinels=[-999])
    print("Missing values after standardization:", standardized["brightness"].isna().sum())
    print("Mean while ignoring missing values:", standardized["brightness"].mean())

    # Lesson 3: mark local outliers as missing while keeping their timestamps.
    section("3. Detect local outliers")
    outlier_mask = rolling_mad_outliers(standardized["brightness"])
    print("Detected outliers:", int(outlier_mask.sum()))
    plot_raw_and_outliers(
        standardized, outlier_mask, OUTPUT_DIR / "01_outlier_detection.png"
    )
    cleaned = standardized.copy()
    cleaned.loc[outlier_mask, "brightness"] = np.nan

    # Lesson 4: interpolate short gaps and preserve long gaps.
    section("4. Interpolate small gaps")
    before_count = int(cleaned["brightness"].isna().sum())
    interpolated = interpolate_small_gaps(cleaned, "brightness", limit=5)
    after_count = int(interpolated["brightness"].isna().sum())
    print("Missing before interpolation:", before_count)
    print("Missing after interpolation:", after_count)
    print("The long gap remains missing:", bool(interpolated.iloc[355:377]["brightness"].isna().all()))
    interpolated = interpolated.rename(columns={"brightness": "brightness_clean"})

    # Lesson 5: compare normalization methods without hiding the retained gap.
    section("5. Shift and scale")
    normalized = add_normalizations(interpolated, "brightness_clean")
    print("Z-score mean:", round(normalized["z_score"].mean(), 12))
    print("Z-score population standard deviation:", round(normalized["z_score"].std(ddof=0), 12))

    # Lesson 6: exclude transit regions when fitting the long-term trend.
    section("6. Remove the unwanted trend")
    time = normalized["time_days"]
    baseline_mask = ~(
        time.between(1.80, 2.20) | time.between(3.98, 4.32)
    )
    detrended = polynomial_detrend(
        normalized, "relative_brightness", degree=3, mask=baseline_mask
    )
    print("Detrended baseline mean:", round(detrended.loc[baseline_mask, "detrended"].mean(), 8))

    # Lesson 7: compare centered and causal time windows.
    section("7. Smooth noisy data")
    detrended["smoothed"] = time_window_smooth(
        detrended, "detrended", before_days=0.05, after_days=0.05, method="mean"
    )
    detrended["trailing_median"] = time_window_smooth(
        detrended, "detrended", before_days=0.10, after_days=0, method="median"
    )
    print("Centered window uses past and future values.")
    print("Trailing window uses present and past values only.")

    # Lesson 8: validate invariants before exporting the prepared dataset.
    section("8. Validate and export")
    assert len(detrended) == len(raw)
    assert detrended["time_days"].is_monotonic_increasing
    assert not (detrended["brightness_clean"] == -999).any()
    assert detrended.iloc[355:377]["brightness_clean"].isna().all()
    assert abs(detrended["z_score"].mean()) < 1e-10
    assert abs(detrended["z_score"].std(ddof=0) - 1) < 1e-10

    export_columns = [
        "time_days",
        "brightness_clean",
        "relative_brightness",
        "trend",
        "detrended",
        "smoothed",
        "trailing_median",
    ]
    final_data = detrended.loc[:, export_columns]
    final_data.to_csv(OUTPUT_DIR / "prepared_light_curve.csv", index=False)
    plot_processing_stages(final_data, OUTPUT_DIR / "02_processing_stages.png")

    # A small audit table makes each transformation reviewable.
    audit = pd.DataFrame(
        {
            "check": [
                "input_rows",
                "sentinel_codes_replaced",
                "outliers_marked_missing",
                "missing_after_short_gap_fill",
                "remaining_long_gap_rows",
            ],
            "value": [
                len(raw),
                int((raw["brightness"] == -999).sum()),
                int(outlier_mask.sum()),
                int(final_data["brightness_clean"].isna().sum()),
                int(final_data.iloc[355:377]["brightness_clean"].isna().sum()),
            ],
        }
    )
    audit.to_csv(OUTPUT_DIR / "quality_audit.csv", index=False)
    print(audit.to_string(index=False))
    print(f"\nCourse outputs saved to {OUTPUT_DIR}")
    return final_data


# Run all lessons only when this file is executed directly.
if __name__ == "__main__":
    run_course()
