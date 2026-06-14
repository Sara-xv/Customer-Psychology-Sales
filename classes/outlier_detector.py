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
ALL_METHODS        = frozenset({"iqr", "zscore", "modified_z", "isolation"})

# Cohesive color palette — mirrors MissingValuesAnalyzer severity scale
COLOR_HEADER  = "#2E4057"   # Deep slate blue  (titles / headers)
COLOR_HIGH    = "#D72631"   # Strong red        (>10 % outliers)
COLOR_MEDIUM  = "#F39C12"   # Amber / orange    (5–10 %)
COLOR_LOW     = "#F7DC6F"   # Soft yellow       (1–5 %)
COLOR_MINOR   = "#3498DB"   # Steel blue        (<1 %)
COLOR_GOOD    = "#27AE60"   # Green             (clean / no outliers)

SEVERITY_THRESHOLDS: List[Tuple[float, str, str]] = [
    (10.0, COLOR_HIGH,   "🔴 HIGH"),
    (5.0,  COLOR_MEDIUM, "🟠 MEDIUM"),
    (1.0,  COLOR_LOW,    "🟡 LOW"),
    (0.0,  COLOR_MINOR,  "🔵 MINOR"),
]


def _severity(pct: float) -> Tuple[str, str]:
    """Return (rich_color, label) for a given outlier percentage."""
    for threshold, color, label in SEVERITY_THRESHOLDS:
        if pct > threshold:
            return color, label
    return SEVERITY_THRESHOLDS[-1][1], SEVERITY_THRESHOLDS[-1][2]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OutlierDetector:
    """
    Detect, report, visualise, and handle outliers in a DataFrame.

    Supported methods
    -----------------
    iqr          : Tukey fences  (Q1 ± 1.5·IQR) — general-purpose
    zscore       : Standard Z-score  (threshold = 3σ) — assumes normality
    modified_z   : Median-based Z-score using MAD — robust to skew/outliers
    isolation    : Isolation Forest — multivariate / high-dimensional data

    Parameters
    ----------
    data : pd.DataFrame
        Input data.  A defensive copy is stored as ``original_data``.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        self.original_data     = data.copy()
        self.data              = data.copy()
        self.console           = Console()
        self.numeric_columns: List[str] = (
            self.data.select_dtypes(include=[np.number]).columns.tolist()
        )

        # Color aliases (consistent with MissingValuesAnalyzer)
        self.color_header = COLOR_HEADER
        self.color_high   = COLOR_HIGH
        self.color_medium = COLOR_MEDIUM
        self.color_low    = COLOR_LOW
        self.color_minor  = COLOR_MINOR
        self.color_good   = COLOR_GOOD

        if not self.numeric_columns:
            self.console.print(
                f"[{COLOR_MEDIUM}]⚠ Warning: No numeric columns found in dataset[/{COLOR_MEDIUM}]"
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
                f"[{COLOR_MEDIUM}]⚠ Column '{column}' is not numeric — skipping[/{COLOR_MEDIUM}]"
            )
            return False
        return True

    @staticmethod
    def _empty_result(method: str) -> Dict:
        return {
            "method": method, "outlier_count": 0,
            "percentage": 0.0, "indices": [], "values": [],
        }

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
        """
        IQR (Tukey Fences) method.

        Flags values outside  Q1 − 1.5·IQR  /  Q3 + 1.5·IQR.
        Best for: symmetric or mildly skewed distributions.
        """
        if not self._validate_column(column):
            return self._empty_result("IQR")

        series = self.data[column].dropna()
        if series.empty:
            return self._empty_result("IQR")

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr     = q3 - q1
        lo, hi  = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        mask = (series < lo) | (series > hi)
        return {
            "method":        "IQR",
            "column":        column,
            "outlier_count": int(mask.sum()),
            "percentage":    mask.sum() / len(series) * 100,
            "lower_bound":   float(lo),
            "upper_bound":   float(hi),
            "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "indices":       series[mask].index.tolist(),
            "values":        series[mask].tolist(),
        }

    def detect_outliers_zscore(
        self, column: str, threshold: float = 3.0
    ) -> Dict:
        """
        Standard Z-Score method.

        Assumes approximate normality.  Values with |z| > threshold are flagged.
        Best for: roughly Gaussian, no heavy tails.
        """
        if not self._validate_column(column):
            return self._empty_result("Z-Score")

        series = self.data[column].dropna()
        if len(series) < 2:
            return self._empty_result("Z-Score")

        std = series.std()
        if std == 0:
            return self._empty_result("Z-Score")

        z    = np.abs((series - series.mean()) / std)
        mask = z > threshold
        return {
            "method":        "Z-Score",
            "column":        column,
            "outlier_count": int(mask.sum()),
            "percentage":    mask.sum() / len(series) * 100,
            "threshold":     threshold,
            "mean":          float(series.mean()),
            "std":           float(std),
            "indices":       series[mask].index.tolist(),
            "values":        series[mask].tolist(),
        }

    def detect_outliers_modified_zscore(
        self, column: str, threshold: float = 3.5
    ) -> Dict:
        """
        Modified Z-Score (Iglewicz & Hoaglin, 1993).

        Uses MAD instead of std — robust against skewed distributions and
        extreme values in the reference set.
        Best for: skewed data, non-normal distributions.
        """
        if not self._validate_column(column):
            return self._empty_result("Modified Z-Score")

        series = self.data[column].dropna()
        if len(series) < 2:
            return self._empty_result("Modified Z-Score")

        median = series.median()
        mad    = np.median(np.abs(series - median))

        if mad == 0:
            return {
                **self._empty_result("Modified Z-Score"),
                "note": "MAD = 0 — all values are identical or distribution is degenerate",
            }

        mz   = 0.6745 * (series - median) / mad
        mask = np.abs(mz) > threshold
        return {
            "method":        "Modified Z-Score",
            "column":        column,
            "outlier_count": int(mask.sum()),
            "percentage":    mask.sum() / len(series) * 100,
            "threshold":     threshold,
            "median":        float(median),
            "mad":           float(mad),
            "indices":       series[mask].index.tolist(),
            "values":        series[mask].tolist(),
        }

    def detect_outliers_isolation_forest(
        self,
        columns: Optional[List[str]] = None,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> Dict:
        """
        Isolation Forest — multivariate outlier detection.

        Suitable for high-dimensional data; does not assume any distribution.
        Particularly effective when outliers form clusters or have complex
        multivariate patterns invisible to univariate methods.

        Parameters
        ----------
        columns       : Columns to use (default = all numeric).
        contamination : Expected fraction of outliers  [0.01, 0.50].
        random_state  : Seed for reproducibility.
        """
        cols = [c for c in (columns or self.numeric_columns)
                if c in self.numeric_columns]

        if not cols:
            return self._empty_result("Isolation Forest")

        df = self.data[cols].dropna()
        if len(df) < 10:
            return {
                **self._empty_result("Isolation Forest"),
                "note": "Insufficient data — need ≥ 10 complete rows",
            }

        contamination = float(np.clip(contamination, 0.01, 0.50))
        preds = IsolationForest(
            contamination=contamination, random_state=random_state
        ).fit_predict(df)

        mask = preds == -1
        return {
            "method":         "Isolation Forest",
            "columns_used":   cols,
            "outlier_count":  int(mask.sum()),
            "percentage":     mask.sum() / len(df) * 100,
            "contamination":  contamination,
            "indices":        df[mask].index.tolist(),
            "values":         df[mask].to_dict("records"),
        }

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def get_outlier_summary(self, method: str = "modified_z") -> pd.DataFrame:
        """
        Per-column outlier summary.

        Parameters
        ----------
        method : {'iqr', 'zscore', 'modified_z'}

        Returns
        -------
        pd.DataFrame with columns:
            Column, Method, Outlier_Count, Outlier_%, Has_Outliers, Severity
        """
        if method not in UNIVARIATE_METHODS:
            raise ValueError(f"method must be one of {sorted(UNIVARIATE_METHODS)}")
        if not self.numeric_columns:
            return pd.DataFrame()

        rows = []
        for col in self.numeric_columns:
            res = self._run_univariate(col, method)
            _, sev_label = _severity(res["percentage"])
            rows.append({
                "Column":        col,
                "Method":        res["method"],
                "Outlier_Count": res["outlier_count"],
                "Outlier_%":     round(res["percentage"], 2),
                "Has_Outliers":  res["outlier_count"] > 0,
                "Severity":      sev_label if res["outlier_count"] > 0 else "✅ Clean",
            })

        return (
            pd.DataFrame(rows)
            .sort_values("Outlier_%", ascending=False)
            .reset_index(drop=True)
        )

    def get_outlier_rows(self, method: str = "modified_z") -> pd.DataFrame:
        """
        Tidy DataFrame of every detected outlier.

        Columns: ``Column``, ``Index``, ``Value``, ``Method``.
        """
        if method not in UNIVARIATE_METHODS:
            raise ValueError(f"method must be one of {sorted(UNIVARIATE_METHODS)}")

        records = [
            {"Column": col, "Index": idx, "Value": val, "Method": res["method"]}
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
        Print a full formatted outlier report to the console.

        Parameters
        ----------
        method : {'iqr', 'zscore', 'modified_z', 'isolation'}
        """
        if method not in ALL_METHODS:
            raise ValueError(f"method must be one of {sorted(ALL_METHODS)}")

        c = self.console
        c.print(
            f"\n[bold {COLOR_HEADER}]📊 OUTLIER DETECTION REPORT[/bold {COLOR_HEADER}]"
        )
        c.print(f"[dim]{'='*70}[/dim]")
        c.print(
            f"  Dataset : {self.data.shape[0]:,} rows × {self.data.shape[1]} columns"
            f"   |   Numeric columns : {len(self.numeric_columns)}"
            f"   |   Method : [bold]{method.replace('_', ' ').title()}[/bold]"
        )
        c.print(f"[dim]{'='*70}[/dim]\n")

        if method == "isolation":
            self._display_isolation_result()
        else:
            self._display_univariate_result(method)

        self._display_recommendations(method)
        c.print(f"\n[dim]{'='*70}[/dim]\n")

    def _display_isolation_result(self) -> None:
        result = self.detect_outliers_isolation_forest()
        color, label = _severity(result["percentage"])
        self.console.print(
            f"  [{color}]Outliers found : {result['outlier_count']:,}"
            f"  ({result['percentage']:.2f}%)   {label}[/{color}]"
        )
        if result.get("note"):
            self.console.print(f"  [{COLOR_MEDIUM}]{result['note']}[/{COLOR_MEDIUM}]")

    def _display_univariate_result(self, method: str) -> None:
        summary = self.get_outlier_summary(method)
        if summary.empty:
            self.console.print(f"[{COLOR_MEDIUM}]No numeric columns to analyse[/{COLOR_MEDIUM}]")
            return

        # ── Overall stats ──────────────────────────────────────────────
        total   = int(summary["Outlier_Count"].sum())
        n_cols  = len(summary)
        n_with  = int(summary["Has_Outliers"].sum())

        stats = Table(
            title="📈 Overall Statistics", box=ROUNDED,
            title_style=f"bold {COLOR_HEADER}",
            header_style=f"bold {COLOR_HEADER}",
        )
        stats.add_column("Metric",  style="bold", justify="left")
        stats.add_column("Value",   justify="right")

        overall_pct = (total / (len(self.data) * n_cols) * 100) if n_cols else 0
        for label, value in [
            ("Total numeric columns",      f"{n_cols}"),
            ("Columns with outliers",      f"{n_with} out of {n_cols}"),
            ("Total outlier count",        f"{total:,}"),
            ("Overall outlier percentage", f"{overall_pct:.2f}%"),
            ("Avg outliers per column",    f"{total / n_cols:.2f}" if n_cols else "—"),
        ]:
            stats.add_row(label, value)
        self.console.print(stats)

        # ── Per-column breakdown ───────────────────────────────────────
        affected = summary[summary["Has_Outliers"]]
        if affected.empty:
            self.console.print(f"\n[{COLOR_GOOD}]✅ No outliers detected in any column![/{COLOR_GOOD}]")
            return

        table = Table(
            title="🎯 Outliers by Column", box=ROUNDED,
            title_style=f"bold {COLOR_HEADER}",
            header_style=f"bold {COLOR_HEADER}",
        )
        for header, kw in [
            ("#",          {"style": "dim", "justify": "center"}),
            ("Column",     {"style": "bold"}),
            ("Count",      {"justify": "right"}),
            ("Percentage", {"justify": "right"}),
            ("Severity",   {"justify": "center"}),
        ]:
            table.add_column(header, **kw)

        for i, (_, row) in enumerate(affected.iterrows(), 1):
            color, label = _severity(row["Outlier_%"])
            table.add_row(
                str(i),
                row["Column"],
                f"[{color}]{int(row['Outlier_Count']):,}[/{color}]",
                f"[{color}]{row['Outlier_%']:.2f}%[/{color}]",
                f"[{color}]{label}[/{color}]",
            )
        self.console.print(table)

    def _display_recommendations(self, method: str = "modified_z") -> None:
        """
        Display professional, context-aware recommendations based on
        severity of detected outliers and available handling strategies.
        """
        summary = self.get_outlier_summary(method) if method != "isolation" else None

        rec_table = Table(
            title="💡 Recommendations", box=ROUNDED,
            title_style=f"bold {COLOR_HEADER}",
            header_style=f"bold {COLOR_HEADER}",
        )
        rec_table.add_column("Strategy",    style="bold", justify="left")
        rec_table.add_column("Severity",    justify="center")
        rec_table.add_column("Best For",    justify="left")
        rec_table.add_column("Snippet",     style="dim", justify="left")

        for strategy, sev, best_for, snippet in [
            (
                "Keep & use robust models",
                f"[{COLOR_MINOR}]Any[/{COLOR_MINOR}]",
                "When outliers are real, meaningful data points",
                "RandomForest, XGBoost, HuberRegressor",
            ),
            (
                "Winsorising / Capping",
                f"[{COLOR_LOW}]LOW–MEDIUM[/{COLOR_LOW}]",
                "Preserve row count; reduce extreme influence on model",
                "series.clip(lower=q1-1.5*iqr, upper=q3+1.5*iqr)",
            ),
            (
                "Log / √ transform",
                f"[{COLOR_LOW}]LOW–MEDIUM[/{COLOR_LOW}]",
                "Right-skewed distributions (income, counts, prices)",
                "np.log1p(col)  /  np.sqrt(col)",
            ),
            (
                "Median imputation",
                f"[{COLOR_MEDIUM}]MEDIUM[/{COLOR_MEDIUM}]",
                "Replace outliers while keeping all rows intact",
                "col.where(~mask, col.median())",
            ),
            (
                "Row removal",
                f"[{COLOR_MEDIUM}]LOW (<1%)[/{COLOR_MEDIUM}]",
                "Only if outliers are data-entry errors or very rare",
                "df.drop(index=outlier_indices)",
            ),
            (
                "Isolation Forest",
                f"[{COLOR_HIGH}]HIGH / Multivariate[/{COLOR_HIGH}]",
                "Complex multivariate outliers invisible to univariate methods",
                "detector.detect_outliers_isolation_forest()",
            ),
            (
                "Separate model / subgroup",
                f"[{COLOR_HIGH}]HIGH[/{COLOR_HIGH}]",
                "Outliers belong to a different regime (e.g., fraud, rare events)",
                "Train separate model on flagged subset",
            ),
        ]:
            rec_table.add_row(strategy, sev, best_for, snippet)

        self.console.print(rec_table)

        # ── Column-level action hints ──────────────────────────────────
        if summary is not None and not summary.empty:
            affected = summary[summary["Has_Outliers"]]
            if not affected.empty:
                hint_table = Table(
                    title="🔎 Column-Level Action Hints", box=ROUNDED,
                    title_style=f"bold {COLOR_HEADER}",
                    header_style=f"bold {COLOR_HEADER}",
                )
                hint_table.add_column("Column",     style="bold")
                hint_table.add_column("Outlier %",  justify="center")
                hint_table.add_column("Severity",   justify="center")
                hint_table.add_column("Suggested Action", justify="left")

                for _, row in affected.iterrows():
                    pct   = row["Outlier_%"]
                    color, label = _severity(pct)

                    if pct > 10:
                        action = ("Do NOT remove blindly — investigate distribution; "
                                  "consider robust model or subgroup analysis")
                    elif pct > 5:
                        action = ("Winsorise to IQR bounds or apply log-transform; "
                                  "verify outliers are not data-entry errors")
                    elif pct > 1:
                        action = ("Cap or impute with median; "
                                  "add outlier indicator column if modeling")
                    else:
                        action = "Safe to remove rows or cap — low impact on dataset size"

                    hint_table.add_row(
                        row["Column"],
                        f"[{color}]{pct:.2f}%[/{color}]",
                        f"[{color}]{label}[/{color}]",
                        action,
                    )
                self.console.print(hint_table)

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
        Boxplot + distribution plot for up to *max_cols* columns.

        Parameters
        ----------
        column   : Focus on a single column (optional).
        method   : Detection method used to highlight outliers.
        max_cols : Maximum number of columns shown when *column* is not given.
        """
        if not self.numeric_columns:
            self.console.print(f"[{COLOR_MEDIUM}]No numeric columns to plot[/{COLOR_MEDIUM}]")
            return

        if column:
            if column not in self.numeric_columns:
                self.console.print(
                    f"[{COLOR_HIGH}]Column '{column}' not found or not numeric[/{COLOR_HIGH}]"
                )
                return
            cols = [column]
        else:
            summary      = self.get_outlier_summary(method)
            with_outliers = summary[summary["Outlier_Count"] > 0]["Column"].tolist()
            cols         = (with_outliers or self.numeric_columns)[:max_cols]

        n           = len(cols)
        ncols_grid  = min(2, n)
        nrows_grid  = (n + 1) // 2

        fig, axes = plt.subplots(
            nrows_grid, ncols_grid,
            figsize=(6 * ncols_grid, 4 * nrows_grid),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for i, col in enumerate(cols):
            ax     = axes_flat[i]
            series = self.data[col].dropna()

            bp = ax.boxplot(series, vert=True, patch_artist=True)
            bp["boxes"][0].set_facecolor("#AED6F1")
            bp["boxes"][0].set_alpha(0.8)
            ax.set_title(f"Boxplot: {col}", fontsize=11, fontweight="bold")
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)

            det_method = method if method != "isolation" else "modified_z"
            res = self._run_univariate(col, det_method)
            if res["outlier_count"]:
                ax.scatter(
                    [1] * res["outlier_count"], res["values"],
                    color="#D72631", s=55, alpha=0.8, zorder=3,
                    label=f"{res['outlier_count']} outliers",
                )
                ax.legend(fontsize=8)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f"Outlier Detection — {method.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

        # ── Distribution plot ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 5))
        for col in cols:
            sns.histplot(
                self.data[col].dropna(), kde=True,
                label=col, alpha=0.5, ax=ax,
            )
        ax.set_title("Distribution of Selected Columns", fontsize=12, fontweight="bold")
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
                f"[{COLOR_GOOD}]✅ Removed {len(bad_indices):,} outlier rows "
                f"({len(bad_indices)/len(self.data)*100:.2f}% of dataset)[/{COLOR_GOOD}]"
            )
            return self.data.drop(index=list(bad_indices)).reset_index(drop=True)

        self.console.print(f"[{COLOR_MEDIUM}]No outliers found to remove[/{COLOR_MEDIUM}]")
        return self.data.copy()

    def cap_outliers(
        self,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Winsorise outliers to the IQR bounds (Q1 − 1.5·IQR / Q3 + 1.5·IQR).

        Returns a new DataFrame (``self.data`` is unchanged).
        """
        cols = [c for c in (columns or self.numeric_columns)
                if c in self.numeric_columns]

        capped = self.data.copy()
        for col in cols:
            res = self.detect_outliers_iqr(col)
            if "lower_bound" in res:
                capped[col] = capped[col].clip(res["lower_bound"], res["upper_bound"])

        self.console.print(
            f"[{COLOR_GOOD}]✅ Winsorised outliers in {len(cols)} column(s)[/{COLOR_GOOD}]"
        )
        return capped

    def impute_outliers_with_median(
        self,
        method: str = "modified_z",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Replace detected outliers with the column median.

        Useful when you want to keep all rows but reduce extreme influence.
        Returns a new DataFrame (``self.data`` is unchanged).
        """
        cols = [c for c in (columns or self.numeric_columns)
                if c in self.numeric_columns]

        imputed = self.data.copy()
        for col in cols:
            res    = self._run_univariate(col, method)
            median = imputed[col].median()
            imputed.loc[res["indices"], col] = median

        self.console.print(
            f"[{COLOR_GOOD}]✅ Imputed outliers with median in {len(cols)} column(s)[/{COLOR_GOOD}]"
        )
        return imputed

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
        Display outliers for each column separately, with full context rows.

        Parameters
        ----------
        method        : {'iqr', 'zscore', 'modified_z'}
        columns       : Columns to analyze (default = all numeric).
        sample_size   : Number of rows to display per column.
        random_sample : If True, sample randomly; else take first N rows.

        Returns
        -------
        Dict[str, pd.DataFrame]
            All outlier rows per column (not just the sample).
        """
        if method not in UNIVARIATE_METHODS:
            raise ValueError(f"method must be one of {sorted(UNIVARIATE_METHODS)}")
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than zero")

        cols = [c for c in (columns or self.numeric_columns)
                if c in self.numeric_columns]

        results: Dict[str, pd.DataFrame] = {}

        with pd.option_context(
            "display.max_columns", None,
            "display.max_rows",    None,
            "display.width",       None,
            "display.max_colwidth", 50,
        ):
            for col in cols:
                res = self._run_univariate(col, method)
                if res["outlier_count"] == 0:
                    continue

                outlier_rows = self.data.reindex(res["indices"]).copy()

                if "Outlier_Value" in outlier_rows.columns:
                    outlier_rows.drop(columns=["Outlier_Value"], inplace=True)
                outlier_rows.insert(0, "Outlier_Value", outlier_rows[col])

                total_outliers = len(outlier_rows)
                display_df = (
                    outlier_rows.sample(n=sample_size, random_state=42)
                    if total_outliers > sample_size and random_sample
                    else outlier_rows.head(sample_size)
                    if total_outliers > sample_size
                    else outlier_rows
                )

                results[col] = outlier_rows

                color, severity = _severity(res["percentage"])
                separator = "─" * 90
                print(f"\n{separator}")
                print(
                    f"📌 Column     : {col}\n"
                    f"📊 Outliers   : {total_outliers:,}  ({res['percentage']:.2f}%)\n"
                    f"🎯 Severity   : {severity}\n"
                    f"👀 Displaying : {len(display_df)} of {total_outliers} rows"
                )
                print(separator)
                print(display_df.to_string())
                print()

        if not results:
            print(f"\n[{COLOR_GOOD}]✅ No outliers detected in any column.[/{COLOR_GOOD}]\n")

        return results