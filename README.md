# Tables

Study materials and runnable examples for working with tabular data in MATLAB and Python. The project covers importing CSV files, calculating league standings, joining tables, preparing data types, and plotting relationships.

## Project contents

| File or folder | Purpose |
| --- | --- |
| `determineEPLstandings.mlx` | MATLAB Live Script for calculating English Premier League standings. |
| `EPLanalysis.mlx` | MATLAB Live Script for joining standings with team information and exploring manager and payroll relationships. |
| `EPLresults.csv` | Home and away statistics for the 2015–16 EPL season. |
| `EPLteams.csv` | Team information, including payroll and manager details. |
| `EPLFinalStandings.csv` | Reference standings used to check the Python calculation. |
| `Python_Tables_Study.py` | Python workflow that builds and validates standings, joins team information, and exports reports and plots. |
| `Python_Tables_Course.ipynb` | Interactive pandas course with exercises and solutions. |
| `Python_Tables_Study_Guide.pdf` / `.tex` | Python tables study guide and editable LaTeX source. |
| `1_Tables.docx` | Tables teaching material. |
| `Tables Quick Reference.pdf` | Tables reference material. |

## Get started

```bash
git clone https://github.com/VICTORBUITHANG/Tables.git
cd Tables
```

### MATLAB

Open the repository folder as the MATLAB current folder so the Live Scripts can find the CSV files. Open and run `determineEPLstandings.mlx`, then explore `EPLanalysis.mlx`.

The examples use MATLAB tables and Live Editor features. Use a MATLAB release that supports the functions used in the scripts, including `donutchart` for the analysis example.

### Python

The scripts use **NumPy**, **pandas**, and **Matplotlib**. Interactive notebooks also require **JupyterLab**. Install them in your chosen Python environment:

```bash
python -m pip install numpy pandas matplotlib jupyterlab
```

For this project's configured local environment, run the tables workflow from the repository root:

```bash
/Users/victorbui/venvs/ai312/bin/python Python_Tables_Study.py
```

On another machine, use `python Python_Tables_Study.py` with the dependencies installed. The script resolves input and output paths relative to its own location.

To use the interactive tables course, launch JupyterLab from the repository root and run the notebook cells in order:

```bash
/Users/victorbui/venvs/ai312/bin/jupyter lab Python_Tables_Course.ipynb
```

## Tables workflow and outputs

The Python script combines home and away statistics, awards three points per win and one per draw, and sorts teams by points and goal difference. It checks the calculated standings against `EPLFinalStandings.csv`, validates team joins, parses manager hire dates, and records payroll in millions of GBP.

Running the script writes these files to `Python_Tables_Output/`:

- `PythonFinalStandings.csv`
- `PythonTeamReport.csv`
- `payroll_vs_points.png`
- `goals_conceded_vs_points.png`

These plots describe associations in the supplied season data; they do not establish causation.

## Generated files

The Python tables workflow recreates its named outputs when run. Generated output folders, notebook checkpoints, Python caches, local environment files, and LaTeX build intermediates are excluded from Git by `.gitignore`. The supplied input CSV files and study documents are tracked.
