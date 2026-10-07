# Python_Tables_Study.py

"""Build league standings and a team analysis from the supplied CSV files."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# Resolve input paths from this script so it works from any current directory.
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_DIR / "EPLresults.csv"
TEAMS_PATH = PROJECT_DIR / "EPLteams.csv"
REFERENCE_PATH = PROJECT_DIR / "EPLFinalStandings.csv"
OUTPUT_DIR = PROJECT_DIR / "Python_Tables_Output"


def build_standings(results):
    """Return ranked team totals from home and away records.

    The input has one row per team and paired Home/Away numeric columns.
    Points use three per win and one per draw; goal difference breaks ties.
    """
    # Start a separate table to keep the imported records intact.
    standings = pd.DataFrame({"Team": results["Team"]})

    # Add each home and away pair by its statistic name.
    for statistic in ("Wins", "Draws", "Losses", "GF", "GA"):
        standings[f"Total{statistic}"] = (
            results[f"Home {statistic}"] + results[f"Away {statistic}"]
        )

    # Calculate ranking fields and keep every team's values together while sorting.
    standings["GoalDiff"] = standings["TotalGF"] - standings["TotalGA"]
    standings["Points"] = 3 * standings["TotalWins"] + standings["TotalDraws"]
    return standings.sort_values(
        ["Points", "GoalDiff"], ascending=[False, False]
    ).reset_index(drop=True)


def prepare_analysis(standings, teams):
    """Join one team-information row to each standing and clean data types.

    Inputs must have unique Team keys. The function raises if a team has no
    match or a hire date does not follow the day-month-year format.
    """
    # Validate the key before joining to prevent duplicate or missing team rows.
    assert standings["Team"].is_unique
    assert teams["Team"].is_unique
    analysis = standings.merge(
        teams, on="Team", how="left", validate="one_to_one", indicator=True
    )
    assert analysis["_merge"].eq("both").all()
    analysis = analysis.drop(columns="_merge")

    # Retain payroll units in its name and parse dates with the known format.
    analysis = analysis.rename(
        columns={
            "Payroll (M£)": "Payroll_MGBP",
            "Manager Hire Date": "ManagerHireDate",
            "Manager Nationality": "ManagerNationality",
        }
    )
    analysis["ManagerHireDate"] = pd.to_datetime(
        analysis["ManagerHireDate"], format="%d %b %Y", errors="raise"
    )
    analysis["ManagerNationality"] = analysis["ManagerNationality"].astype(
        "category"
    )
    return analysis


def save_scatter(data, x_column, x_label, title, destination):
    """Save a descriptive scatter plot of one numeric field against points.

    The destination is a PNG path. The function writes the image and closes
    its figure, so repeated calls do not leave figures open.
    """
    # Label both axes so the variable and payroll units remain clear.
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(data[x_column], data["Points"])
    axis.set(xlabel=x_label, ylabel="League points", title=title)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    """Read the supplied CSVs, validate standings, and save tables and plots.

    Outputs are written to Python_Tables_Output beside this script. The source
    files are read only and remain unchanged.
    """
    # Load the supplied season records and reproduce the ranked standings.
    results = pd.read_csv(RESULTS_PATH)
    teams = pd.read_csv(TEAMS_PATH)
    standings = build_standings(results)
    assert (
        standings["TotalWins"]
        + standings["TotalDraws"]
        + standings["TotalLosses"]
        == 38
    ).all()
    reference = pd.read_csv(REFERENCE_PATH)
    pd.testing.assert_frame_equal(standings, reference, check_dtype=False)

    # Join team information and create a compact, exportable report.
    analysis = prepare_analysis(standings, teams)
    report_columns = [
        "Team", "Points", "GoalDiff", "Payroll_MGBP", "ManagerNationality"
    ]
    report = analysis.loc[:, report_columns]
    assert len(report) == 20 and report["Team"].is_unique
    assert not report.isna().any().any()

    # Export two CSVs and two plots for review or further study.
    OUTPUT_DIR.mkdir(exist_ok=True)
    standings.to_csv(OUTPUT_DIR / "PythonFinalStandings.csv", index=False)
    report.to_csv(OUTPUT_DIR / "PythonTeamReport.csv", index=False)
    save_scatter(
        analysis, "Payroll_MGBP", "Payroll (million GBP)",
        "Payroll and points, 2015–16", OUTPUT_DIR / "payroll_vs_points.png"
    )
    save_scatter(
        analysis, "TotalGA", "Goals conceded",
        "Goals conceded and points, 2015–16",
        OUTPUT_DIR / "goals_conceded_vs_points.png"
    )
    print(f"Saved tables and plots to {OUTPUT_DIR}")
    print(report.head(5).to_string(index=False))


# Run the workflow only when invoked as a script.
if __name__ == "__main__":
    main()
