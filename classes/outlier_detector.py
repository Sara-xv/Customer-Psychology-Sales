# classes/outlier_detector.py
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.box import ROUNDED
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNIVARIATE_METHODS = frozenset({"iqr", "zscore", "modified_z"})
ALL_METHODS = frozenset({"iqr", "zscore", "modified_z", "isolation"})

SEVERITY_THRESHOLDS: List[Tuple[float, str, str]] = [
    (10.0, "#B22222", "🔴 HIGH"),
    (5.0,  "#FF8C00", "🟠 MEDIUM"),
    (1.0,  "#D2691E", "🟡 LOW"),
    (0.0,  "#4682B4", "🟢 MINOR"),
]

COLOR_HEADER = "#2E8B57"


def _severity(pct: float) -> Tuple[str, str]:
    for threshold, color, label in SEVERITY_THRESHOLDS:
        if pct > threshold:
            return color, label
    return SEVERITY_THRESHOLDS[-1][1], SEVERITY_THRESHOLDS[-1][2]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OutlierDetector:
    """
    Outlier detection with four methods: IQR, Z-Score, Modified Z-Score,
    Isolation Forest.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.  A defensive copy is kept as ``original_data``.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        self.original_data = data.copy()
        self.data = data.copy()
        self.console = Console()
        self.numeric_columns: List[str] = (
            self.data.select_dtypes(include=[np.number]).columns.tolist()
        )

        if not self.numeric_columns:
            self.console.print(
                "[yellow]⚠ Warning: No numeric columns found in dataset[/yellow]"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_column(self, column: str) -> bool:
        """Return True if *column* exists and is numeric, else raise / warn."""
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            self.console.print(
                f"[yellow]⚠ Column '{column}' is not numeric — skipping[/yellow]"
            )
            return False
        return True

    @staticmethod
    def _empty_result(method: str) -> Dict:
        return {"method": method, "outlier_count": 0, "percentage": 0.0,
                "indices": [], "values": []}

    def _run_univariate(self, col: str, method: str) -> Dict:
        """Dispatch to the correct univariate detection method."""
        dispatch = {
            "iqr":        self.detect_outliers_iqr,
            "zscore":     self.detect_outliers_zscore,
            "modified_z": self.detect_outliers_modified_zscore,
        }
        return dispatch[method](col)

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect_outliers_iqr(self, column: str) -> Dict:
        """IQR method: flags values outside Q1 ± 1.5·IQR."""
        if not self._validate_column(column):
            return self._empty_result("IQR")

        series = self.data[column].dropna()
        if series.empty:
            return self._empty_result("IQR")

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        mask = (series < lo) | (series > hi)
        return {
            "method": "IQR",
            "column": column,
            "outlier_count": int(mask.sum()),
            "percentage": mask.sum() / len(series) * 100,
            "lower_bound": float(lo),
            "upper_bound": float(hi),
            "indices": series[mask].index.tolist(),
            "values": series[mask].tolist(),
        }

    def detect_outliers_zscore(
        self, column: str, threshold: float = 3.0
    ) -> Dict:
        """Z-Score method (assumes approximate normality)."""
        if not self._validate_column(column):
            return self._empty_result("Z-Score")

        series = self.data[column].dropna()
        if len(series) < 2:
            return self._empty_result("Z-Score")

        std = series.std()
        if std == 0:
            return self._empty_result("Z-Score")

        z = np.abs((series - series.mean()) / std)
        mask = z > threshold
        return {
            "method": "Z-Score",
            "column": column,
            "outlier_count": int(mask.sum()),
            "percentage": mask.sum() / len(series) * 100,
            "threshold": threshold,
            "indices": series[mask].index.tolist(),
            "values": series[mask].tolist(),
        }

    def detect_outliers_modified_zscore(
        self, column: str, threshold: float = 3.5
    ) -> Dict:
        """Modified Z-Score — robust against skewed distributions (uses MAD)."""
        if not self._validate_column(column):
            return self._empty_result("Modified Z-Score")

        series = self.data[column].dropna()
        if len(series) < 2:
            return self._empty_result("Modified Z-Score")

        median = series.median()
        mad = np.median(np.abs(series - median))

        if mad == 0:
            return {**self._empty_result("Modified Z-Score"),
                    "note": "MAD = 0 (all values identical)"}

        mz = 0.6745 * (series - median) / mad
        mask = np.abs(mz) > threshold
        return {
            "method": "Modified Z-Score",
            "column": column,
            "outlier_count": int(mask.sum()),
            "percentage": mask.sum() / len(series) * 100,
            "threshold": threshold,
            "indices": series[mask].index.tolist(),
            "values": series[mask].tolist(),
        }

    def detect_outliers_isolation_forest(
        self,
        columns: Optional[List[str]] = None,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> Dict:
        """
        Isolation Forest — best for multivariate outliers.

        Parameters
        ----------
        columns : list of str, optional
            Columns to use.  Defaults to all numeric columns.
        contamination : float
            Expected proportion of outliers (clamped to [0.01, 0.50]).
        random_state : int
            Seed for reproducibility.
        """
        cols = [c for c in (columns or self.numeric_columns)
                if c in self.numeric_columns]

        if not cols:
            return self._empty_result("Isolation Forest")

        df = self.data[cols].dropna()
        if len(df) < 10:
            return {**self._empty_result("Isolation Forest"),
                    "note": "Insufficient data (need ≥ 10 rows)"}

        contamination = float(np.clip(contamination, 0.01, 0.50))
        preds = IsolationForest(
            contamination=contamination, random_state=random_state
        ).fit_predict(df)

        mask = preds == -1
        return {
            "method": "Isolation Forest",
            "columns_used": cols,
            "outlier_count": int(mask.sum()),
            "percentage": mask.sum() / len(df) * 100,
            "contamination": contamination,
            "indices": df[mask].index.tolist(),
            "values": df[mask].to_dict("records"),
        }

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def get_outlier_summary(self, method: str = "modified_z") -> pd.DataFrame:
        """
        Return a per-column outlier summary as a DataFrame.

        Parameters
        ----------
        method : {'iqr', 'zscore', 'modified_z'}
        """
        if method not in UNIVARIATE_METHODS:
            raise ValueError(
                f"method must be one of {sorted(UNIVARIATE_METHODS)}"
            )
        if not self.numeric_columns:
            return pd.DataFrame()

        rows = []
        for col in self.numeric_columns:
            res = self._run_univariate(col, method)
            rows.append({
                "Column": col,
                "Method": res["method"],
                "Outlier_Count": res["outlier_count"],
                "Outlier_%": round(res["percentage"], 2),
                "Has_Outliers": res["outlier_count"] > 0,
            })

        return (
            pd.DataFrame(rows)
            .sort_values("Outlier_%", ascending=False)
            .reset_index(drop=True)
        )

    def get_outlier_rows(self, method: str = "modified_z") -> pd.DataFrame:
        """
        Return a tidy DataFrame of every detected outlier row.

        Columns: ``Column``, ``Index``, ``Value``, ``Method``.
        """
        if method not in UNIVARIATE_METHODS:
            raise ValueError(
                f"method must be one of {sorted(UNIVARIATE_METHODS)}"
            )

        records = [
            {"Column": col, "Index": idx, "Value": val,
             "Method": res["method"]}
            for col in self.numeric_columns
            for res in [self._run_univariate(col, method)]
            for idx, val in zip(res["indices"], res["values"])
        ]
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display_report(self, method: str = "modified_z") -> None:
        """
        Print a formatted outlier report to the console.

        Parameters
        ----------
        method : {'iqr', 'zscore', 'modified_z', 'isolation'}
        """
        if method not in ALL_METHODS:
            raise ValueError(f"method must be one of {sorted(ALL_METHODS)}")

        c = self.console
        c.print(
            f"\n[bold {COLOR_HEADER}]📊 OUTLIER DETECTION REPORT"
            f"[/bold {COLOR_HEADER}]"
        )
        c.print("=" * 70)
        c.print(
            f"Dataset: {self.data.shape[0]} rows × {self.data.shape[1]} cols"
            f"   |   Numeric: {len(self.numeric_columns)}"
            f"   |   Method: {method.replace('_', ' ').title()}"
        )

        if method == "isolation":
            self._display_isolation_result()
        else:
            self._display_univariate_result(method)

        self._display_recommendations()
        c.print("=" * 70 + "\n")

    def _display_isolation_result(self) -> None:
        result = self.detect_outliers_isolation_forest()
        self.console.print(
            f"\n  Outliers found : {result['outlier_count']}"
            f"  ({result['percentage']:.2f} %)"
        )
        if result.get("note"):
            self.console.print(f"  [yellow]{result['note']}[/yellow]")

    def _display_univariate_result(self, method: str) -> None:
        summary = self.get_outlier_summary(method)
        if summary.empty:
            self.console.print("[yellow]No numeric columns to analyse[/yellow]")
            return

        # ── overall stats ──────────────────────────────────────────────
        total = int(summary["Outlier_Count"].sum())
        n_cols = len(summary)
        n_with = int(summary["Has_Outliers"].sum())

        stats = Table(
            title="📈 Overall Statistics", box=ROUNDED,
            title_style=f"bold {COLOR_HEADER}"
        )
        stats.add_column("Metric", style="cyan")
        stats.add_column("Value", justify="right")
        for label, value in [
            ("Total numeric columns",   str(n_cols)),
            ("Columns with outliers",   str(n_with)),
            ("Total outlier count",     str(total)),
            ("Avg outliers per column", f"{total / n_cols:.2f}"),
        ]:
            stats.add_row(label, value)
        self.console.print(stats)

        # ── per-column breakdown ───────────────────────────────────────
        affected = summary[summary["Has_Outliers"]]
        if affected.empty:
            self.console.print("[green]✅ No outliers detected![/green]")
            return

        table = Table(
            title="🎯 Outliers by Column", box=ROUNDED, title_style="bold"
        )
        for header, kw in [
            ("#",             {"style": "dim"}),
            ("Column",        {"style": "cyan"}),
            ("Count",         {"justify": "right"}),
            ("Percentage",    {"justify": "right"}),
            ("Severity",      {"justify": "center"}),
        ]:
            table.add_column(header, **kw)

        for i, (_, row) in enumerate(affected.iterrows(), 1):
            color, label = _severity(row["Outlier_%"])
            table.add_row(
                str(i),
                row["Column"],
                f"[{color}]{int(row['Outlier_Count'])}[/{color}]",
                f"[{color}]{row['Outlier_%']:.2f} %[/{color}]",
                label,
            )
        self.console.print(table)

    def _display_recommendations(self) -> None:
        table = Table(
            title="💡 Recommendations", box=ROUNDED,
            title_style="bold green"
        )
        table.add_column("Action",    style="cyan")
        table.add_column("Best For",  style="white")
        table.add_column("Snippet",   style="dim")

        for action, best_for, snippet in [
            ("Capping / Winsorizing",      "1–10 % outliers",              "series.clip(lo, hi)"),
            ("Row removal",                "< 1 %, large dataset",          "df.drop(index=idx)"),
            ("Median imputation",          "Few outliers, keep all rows",   "series.fillna(median)"),
            ("Robust models",              "Many outliers / high-dim data", "RandomForest, XGBoost"),
            ("Log / √ transform",          "Skewed distribution",           "np.log1p() / np.sqrt()"),
        ]:
            table.add_row(action, best_for, snippet)
        self.console.print(table)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_outliers(
        self,
        column: Optional[str] = None,
        method: str = "modified_z",
        max_cols: int = 4,
    ) -> None:
        """
        Boxplot + histogram for up to *max_cols* columns.

        Parameters
        ----------
        column   : focus on a single column (optional).
        method   : detection method used to highlight outliers.
        max_cols : maximum number of columns shown (ignored when *column* given).
        """
        if not self.numeric_columns:
            self.console.print("[yellow]No numeric columns to plot[/yellow]")
            return

        if column:
            if column not in self.numeric_columns:
                self.console.print(
                    f"[red]Column '{column}' not found or not numeric[/red]"
                )
                return
            cols = [column]
        else:
            summary = self.get_outlier_summary(method)
            with_outliers = summary[summary["Outlier_Count"] > 0]["Column"].tolist()
            cols = (with_outliers or self.numeric_columns)[:max_cols]

        n = len(cols)
        ncols_grid = min(2, n)
        nrows_grid = (n + 1) // 2

        fig, axes = plt.subplots(
            nrows_grid, ncols_grid, figsize=(6 * ncols_grid, 4 * nrows_grid),
            squeeze=False
        )
        axes_flat = axes.flatten()

        for i, col in enumerate(cols):
            ax = axes_flat[i]
            series = self.data[col].dropna()

            bp = ax.boxplot(series, vert=True, patch_artist=True)
            bp["boxes"][0].set_facecolor("lightblue")
            bp["boxes"][0].set_alpha(0.7)
            ax.set_title(f"Boxplot: {col}", fontsize=11, fontweight="bold")
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)

            res = self._run_univariate(col, method if method != "isolation" else "modified_z")
            if res["outlier_count"]:
                ax.scatter(
                    [1] * res["outlier_count"], res["values"],
                    color="red", s=50, alpha=0.7, zorder=3,
                    label=f"{res['outlier_count']} outliers",
                )
                ax.legend(fontsize=8)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f"Outlier Detection — {method.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

        # ── distribution plot ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 5))
        for col in cols:
            sns.histplot(
                self.data[col].dropna(), kde=True, label=col, alpha=0.5, ax=ax
            )
        ax.set_title("Distribution of Selected Columns",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Handling
    # ------------------------------------------------------------------

    def remove_outliers(
        self,
        method: str = "modified_z",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Drop every row that is an outlier in *any* of the given columns.

        Returns a new DataFrame (``self.data`` is unchanged).
        """
        cols = [c for c in (columns or self.numeric_columns)
                if c in self.numeric_columns]

        bad_indices: set = set()
        for col in cols:
            bad_indices.update(self._run_univariate(col, method)["indices"])

        if bad_indices:
            self.console.print(
                f"[green]✅ Removed {len(bad_indices)} outlier rows[/green]"
            )
            return self.data.drop(index=list(bad_indices)).reset_index(drop=True)

        self.console.print("[yellow]No outliers found to remove[/yellow]")
        return self.data.copy()

    def cap_outliers(
        self,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Winsorise outliers to the IQR bounds (Q1 − 1.5·IQR / Q3 + 1.5·IQR).

        IQR is used regardless of the detection method because it directly
        yields interpretable lower / upper bounds.

        Returns a new DataFrame (``self.data`` is unchanged).
        """
        cols = [c for c in (columns or self.numeric_columns)
                if c in self.numeric_columns]

        capped = self.data.copy()
        for col in cols:
            res = self.detect_outliers_iqr(col)
            if "lower_bound" in res:
                capped[col] = capped[col].clip(
                    res["lower_bound"], res["upper_bound"]
                )

        self.console.print(
            f"[green]✅ Capped outliers in {len(cols)} column(s)[/green]"
        )
        return capped

    # ------------------------------------------------------------------
    # Per-column outlier display
    # ------------------------------------------------------------------

    def display_outliers_per_column(
        self,
        method: str = "modified_z",
        columns: Optional[List[str]] = None,
        sample_size: int = 20,
        random_sample: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Display outliers for each column separately.

        Parameters
        ----------
        method : {'iqr', 'zscore', 'modified_z'}
            Outlier detection method.

        columns : list[str], optional
            Columns to analyze.
            Default = all numeric columns.

        sample_size : int
            Number of samples to display per column.

        random_sample : bool
            If True, display random samples.
            If False, display first rows.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing ALL outlier rows per column.
        """

        if method not in UNIVARIATE_METHODS:
            raise ValueError(
                f"method must be one of {sorted(UNIVARIATE_METHODS)}"
            )

        if sample_size <= 0:
            raise ValueError(
                "sample_size must be greater than zero"
            )

        cols = [
            c for c in (columns or self.numeric_columns)
            if c in self.numeric_columns
        ]

        results: Dict[str, pd.DataFrame] = {}

        with pd.option_context(
            "display.max_columns", None,
            "display.max_rows", None,
            "display.width", None,
            "display.max_colwidth", 50,
        ):

            for col in cols:

                res = self._run_univariate(col, method)

                if res["outlier_count"] == 0:
                    continue

                # --------------------------------------------------
                # Get ALL outlier rows
                # --------------------------------------------------

                outlier_rows = self.data.reindex(
                    res["indices"]
                ).copy()

                # Add outlier value column
                if "Outlier_Value" in outlier_rows.columns:
                    outlier_rows.drop(
                        columns=["Outlier_Value"],
                        inplace=True
                    )

                outlier_rows.insert(
                    0,
                    "Outlier_Value",
                    outlier_rows[col]
                )

                total_outliers = len(outlier_rows)

                # --------------------------------------------------
                # Select rows for display only
                # --------------------------------------------------

                if total_outliers > sample_size:

                    if random_sample:
                        display_df = outlier_rows.sample(
                            n=sample_size,
                            random_state=42
                        )
                    else:
                        display_df = outlier_rows.head(
                            sample_size
                        )

                else:
                    display_df = outlier_rows

                # Store ALL outliers
                results[col] = outlier_rows

                # --------------------------------------------------
                # Display
                # --------------------------------------------------

                _, severity = _severity(
                    res["percentage"]
                )

                separator = "─" * 90

                print(f"\n{separator}")

                print(
                    f"📌 Column: {col}\n"
                    f"📊 Total Outliers: {total_outliers}\n"
                    f"📈 Percentage: {res['percentage']:.2f}%\n"
                    f"🎯 Severity: {severity}\n"
                    f"👀 Showing: {len(display_df)} of {total_outliers}"
                )

                print(separator)

                print(display_df.to_string())

                print()

        if not results:
            print("\n✅ No outliers detected.\n")

        return results