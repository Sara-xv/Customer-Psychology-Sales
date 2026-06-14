# classes/outlier_detector.py
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.box import ROUNDED
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy import stats as scipy_stats
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNIVARIATE_METHODS = frozenset({"iqr", "zscore", "modified_z"})
ALL_METHODS        = frozenset({"iqr", "zscore", "modified_z", "isolation"})

COLOR_HEADER  = "#2E4057"
COLOR_HIGH    = "#D72631"
COLOR_MEDIUM  = "#F39C12"
COLOR_LOW     = "#F7DC6F"
COLOR_MINOR   = "#3498DB"
COLOR_GOOD    = "#27AE60"

SEVERITY_THRESHOLDS: List[Tuple[float, str, str]] = [
    (10.0, COLOR_HIGH,   "🔴 HIGH"),
    (5.0,  COLOR_MEDIUM, "🟠 MEDIUM"),
    (1.0,  COLOR_LOW,    "🟡 LOW"),
    (0.0,  COLOR_MINOR,  "🔵 MINOR"),
]

# Skewness thresholds for distribution diagnosis
SKEW_HIGH     = 1.5    # highly skewed  → IQR/modified_z may over-flag
SKEW_MODERATE = 0.75   # moderately skewed


def _severity(pct: float) -> Tuple[str, str]:
    for threshold, color, label in SEVERITY_THRESHOLDS:
        if pct > threshold:
            return color, label
    return SEVERITY_THRESHOLDS[-1][1], SEVERITY_THRESHOLDS[-1][2]


def _skew_label(skew: float) -> Tuple[str, str]:
    """Return (label, warning) based on skewness value."""
    abs_skew = abs(skew)
    if abs_skew > SKEW_HIGH:
        direction = "right" if skew > 0 else "left"
        return (
            f"Highly {direction}-skewed ({skew:+.2f})",
            "⚠️  High skew — outlier method may over-flag. Consider log-transform or IQR with wider fence."
        )
    elif abs_skew > SKEW_MODERATE:
        direction = "right" if skew > 0 else "left"
        return (
            f"Moderately {direction}-skewed ({skew:+.2f})",
            "ℹ️  Moderate skew — verify flagged values are genuine outliers before acting."
        )
    else:
        return f"Approx. normal ({skew:+.2f})", ""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OutlierDetector:
    """
    Detect, diagnose, visualise, and handle outliers in a DataFrame.

    Supported methods
    -----------------
    iqr          : Tukey fences (Q1 ± 1.5·IQR) — general-purpose
    zscore       : Standard Z-score (threshold=3σ) — assumes normality
    modified_z   : MAD-based Z-score — robust to skew/outliers
    isolation    : Isolation Forest — multivariate / high-dimensional

    Key features
    ------------
    - Automatic skewness diagnosis per column (warns when method may over-flag)
    - Histogram + KDE + boxplot per column inside plot_distributions()
    - plot_outliers() shows boxplot with flagged points highlighted
    - display_report() includes skewness warnings and per-column action hints
    """

    def __init__(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        self.original_data    = data.copy()
        self.data             = data.copy()
        self.console          = Console()
        self.numeric_columns: List[str] = (
            self.data.select_dtypes(include=[np.number]).columns.tolist()
        )

        self.color_header = COLOR_HEADER
        self.color_high   = COLOR_HIGH
        self.color_medium = COLOR_MEDIUM
        self.color_low    = COLOR_LOW
        self.color_minor  = COLOR_MINOR
        self.color_good   = COLOR_GOOD

        if not self.numeric_columns:
            self.console.print(
                f"[{COLOR_MEDIUM}]⚠ Warning: No numeric columns found[/{COLOR_MEDIUM}]"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_column(self, column: str) -> bool:
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            self.console.print(
                f"[{COLOR_MEDIUM}]⚠ '{column}' is not numeric — skipping[/{COLOR_MEDIUM}]"
            )
            return False
        return True

    @staticmethod
    def _empty_result(method: str) -> Dict:
        return {"method": method, "outlier_count": 0,
                "percentage": 0.0, "indices": [], "values": []}

    def _run_univariate(self, col: str, method: str) -> Dict:
        return {
            "iqr":        self.detect_outliers_iqr,
            "zscore":     self.detect_outliers_zscore,
            "modified_z": self.detect_outliers_modified_zscore,
        }[method](col)

    def _get_distribution_stats(self, column: str) -> Dict:
        """Return skewness, kurtosis, normality test for a column."""
        series = self.data[column].dropna()
        skew   = float(series.skew())
        kurt   = float(series.kurtosis())

        # Shapiro-Wilk (only reliable for n < 5000; use sample otherwise)
        sample = series.sample(min(5000, len(series)), random_state=42)
        _, p_normal = scipy_stats.shapiro(sample)

        skew_lbl, skew_warn = _skew_label(skew)
        return {
            "skewness":      skew,
            "kurtosis":      kurt,
            "skew_label":    skew_lbl,
            "skew_warning":  skew_warn,
            "is_normal":     p_normal > 0.05,
            "normality_p":   float(p_normal),
            "min":  float(series.min()),
            "max":  float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std":  float(series.std()),
        }

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect_outliers_iqr(self, column: str) -> Dict:
        """IQR (Tukey Fences): Q1 − 1.5·IQR / Q3 + 1.5·IQR."""
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
            "method": "IQR", "column": column,
            "outlier_count": int(mask.sum()),
            "percentage": mask.sum() / len(series) * 100,
            "lower_bound": float(lo), "upper_bound": float(hi),
            "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "indices": series[mask].index.tolist(),
            "values": series[mask].tolist(),
        }

    def detect_outliers_zscore(self, column: str, threshold: float = 3.0) -> Dict:
        """Standard Z-Score — assumes approximate normality."""
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
            "method": "Z-Score", "column": column,
            "outlier_count": int(mask.sum()),
            "percentage": mask.sum() / len(series) * 100,
            "threshold": threshold,
            "mean": float(series.mean()), "std": float(std),
            "indices": series[mask].index.tolist(),
            "values": series[mask].tolist(),
        }

    def detect_outliers_modified_zscore(self, column: str, threshold: float = 3.5) -> Dict:
        """Modified Z-Score (Iglewicz & Hoaglin) — uses MAD, robust to skew."""
        if not self._validate_column(column):
            return self._empty_result("Modified Z-Score")
        series = self.data[column].dropna()
        if len(series) < 2:
            return self._empty_result("Modified Z-Score")
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:
            return {**self._empty_result("Modified Z-Score"),
                    "note": "MAD = 0 — distribution is degenerate"}
        mz = 0.6745 * (series - median) / mad
        mask = np.abs(mz) > threshold
        return {
            "method": "Modified Z-Score", "column": column,
            "outlier_count": int(mask.sum()),
            "percentage": mask.sum() / len(series) * 100,
            "threshold": threshold,
            "median": float(median), "mad": float(mad),
            "indices": series[mask].index.tolist(),
            "values": series[mask].tolist(),
        }

    def detect_outliers_isolation_forest(
        self,
        columns: Optional[List[str]] = None,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> Dict:
        """Isolation Forest — multivariate, no distribution assumption."""
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
            "method": "Isolation Forest", "columns_used": cols,
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
        """Per-column outlier summary with skewness info."""
        if method not in UNIVARIATE_METHODS:
            raise ValueError(f"method must be one of {sorted(UNIVARIATE_METHODS)}")
        if not self.numeric_columns:
            return pd.DataFrame()

        rows = []
        for col in self.numeric_columns:
            res   = self._run_univariate(col, method)
            dstat = self._get_distribution_stats(col)
            _, sev_label = _severity(res["percentage"])
            rows.append({
                "Column":        col,
                "Method":        res["method"],
                "Outlier_Count": res["outlier_count"],
                "Outlier_%":     round(res["percentage"], 2),
                "Skewness":      round(dstat["skewness"], 2),
                "Has_Outliers":  res["outlier_count"] > 0,
                "Severity":      sev_label if res["outlier_count"] > 0 else "✅ Clean",
            })

        return (
            pd.DataFrame(rows)
            .sort_values("Outlier_%", ascending=False)
            .reset_index(drop=True)
        )

    def get_outlier_rows(self, method: str = "modified_z") -> pd.DataFrame:
        """Tidy DataFrame of every detected outlier row."""
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
        """Full outlier report with skewness diagnosis and action hints."""
        if method not in ALL_METHODS:
            raise ValueError(f"method must be one of {sorted(ALL_METHODS)}")

        c = self.console
        c.print(f"\n[bold {COLOR_HEADER}]📊 OUTLIER DETECTION REPORT[/bold {COLOR_HEADER}]")
        c.print(f"[dim]{'='*70}[/dim]")
        c.print(
            f"  Dataset : {self.data.shape[0]:,} rows × {self.data.shape[1]} columns"
            f"   |   Numeric : {len(self.numeric_columns)}"
            f"   |   Method : [bold]{method.replace('_', ' ').title()}[/bold]"
        )
        c.print(f"[dim]{'='*70}[/dim]\n")

        if method == "isolation":
            self._display_isolation_result()
        else:
            self._display_univariate_result(method)
            self._display_skewness_diagnosis(method)

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

        total  = int(summary["Outlier_Count"].sum())
        n_cols = len(summary)
        n_with = int(summary["Has_Outliers"].sum())
        overall_pct = (total / (len(self.data) * n_cols) * 100) if n_cols else 0

        stats = Table(title="📈 Overall Statistics", box=ROUNDED,
                      title_style=f"bold {COLOR_HEADER}",
                      header_style=f"bold {COLOR_HEADER}")
        stats.add_column("Metric", style="bold")
        stats.add_column("Value",  justify="right")
        for lbl, val in [
            ("Total numeric columns",      str(n_cols)),
            ("Columns with outliers",      f"{n_with} out of {n_cols}"),
            ("Total outlier count",        f"{total:,}"),
            ("Overall outlier percentage", f"{overall_pct:.2f}%"),
            ("Avg outliers per column",    f"{total/n_cols:.2f}" if n_cols else "—"),
        ]:
            stats.add_row(lbl, val)
        self.console.print(stats)

        affected = summary[summary["Has_Outliers"]]
        if affected.empty:
            self.console.print(f"\n[{COLOR_GOOD}]✅ No outliers detected![/{COLOR_GOOD}]")
            return

        table = Table(title="🎯 Outliers by Column", box=ROUNDED,
                      title_style=f"bold {COLOR_HEADER}",
                      header_style=f"bold {COLOR_HEADER}")
        for header, kw in [
            ("#",          {"style": "dim", "justify": "center"}),
            ("Column",     {"style": "bold"}),
            ("Count",      {"justify": "right"}),
            ("Percentage", {"justify": "right"}),
            ("Skewness",   {"justify": "center"}),
            ("Severity",   {"justify": "center"}),
        ]:
            table.add_column(header, **kw)

        for i, (_, row) in enumerate(affected.iterrows(), 1):
            color, label = _severity(row["Outlier_%"])
            skew = row["Skewness"]
            skew_color = (COLOR_HIGH if abs(skew) > SKEW_HIGH
                          else COLOR_MEDIUM if abs(skew) > SKEW_MODERATE
                          else COLOR_GOOD)
            table.add_row(
                str(i),
                row["Column"],
                f"[{color}]{int(row['Outlier_Count']):,}[/{color}]",
                f"[{color}]{row['Outlier_%']:.2f}%[/{color}]",
                f"[{skew_color}]{skew:+.2f}[/{skew_color}]",
                f"[{color}]{label}[/{color}]",
            )
        self.console.print(table)

    def _display_skewness_diagnosis(self, method: str) -> None:
        """
        Show per-column skewness diagnosis and warn when the chosen
        detection method is likely to produce false positives.
        """
        diag_table = Table(
            title="🔬 Distribution Diagnosis", box=ROUNDED,
            title_style=f"bold {COLOR_HEADER}",
            header_style=f"bold {COLOR_HEADER}",
        )
        diag_table.add_column("Column",       style="bold")
        diag_table.add_column("Min → Max",    justify="center")
        diag_table.add_column("Skewness",     justify="center")
        diag_table.add_column("Normality",    justify="center")
        diag_table.add_column("Diagnosis",    justify="left")

        for col in self.numeric_columns:
            d = self._get_distribution_stats(col)
            skew_color = (COLOR_HIGH if abs(d["skewness"]) > SKEW_HIGH
                          else COLOR_MEDIUM if abs(d["skewness"]) > SKEW_MODERATE
                          else COLOR_GOOD)
            normal_str = (f"[{COLOR_GOOD}]Normal[/{COLOR_GOOD}]" if d["is_normal"]
                          else f"[{COLOR_MEDIUM}]Non-normal[/{COLOR_MEDIUM}]")
            warning = d["skew_warning"]
            diag = (f"[{COLOR_HIGH}]{warning}[/{COLOR_HIGH}]" if warning
                    else f"[{COLOR_GOOD}]✓ Distribution looks healthy[/{COLOR_GOOD}]")

            diag_table.add_row(
                col,
                f"{d['min']:.1f} → {d['max']:.1f}",
                f"[{skew_color}]{d['skewness']:+.2f}[/{skew_color}]",
                normal_str,
                diag,
            )
        self.console.print(diag_table)

    def _display_recommendations(self, method: str = "modified_z") -> None:
        summary = self.get_outlier_summary(method) if method != "isolation" else None

        rec_table = Table(title="💡 General Recommendations", box=ROUNDED,
                          title_style=f"bold {COLOR_HEADER}",
                          header_style=f"bold {COLOR_HEADER}")
        rec_table.add_column("Strategy",  style="bold")
        rec_table.add_column("Severity",  justify="center")
        rec_table.add_column("Best For",  justify="left")
        rec_table.add_column("Snippet",   style="dim")

        for strategy, sev, best_for, snippet in [
            ("Keep & use robust models",
             f"[{COLOR_MINOR}]Any[/{COLOR_MINOR}]",
             "Outliers are real data points (e.g., high spenders)",
             "RandomForest, XGBoost, HuberRegressor"),
            ("Winsorising / Capping",
             f"[{COLOR_LOW}]LOW–MEDIUM[/{COLOR_LOW}]",
             "Preserve all rows; reduce extreme influence",
             "series.clip(q1-1.5*iqr, q3+1.5*iqr)"),
            ("Log / √ transform",
             f"[{COLOR_LOW}]LOW–MEDIUM[/{COLOR_LOW}]",
             "Right-skewed: reduces false outlier flags",
             "np.log1p(col)  /  np.sqrt(col)"),
            ("Median imputation",
             f"[{COLOR_MEDIUM}]MEDIUM[/{COLOR_MEDIUM}]",
             "Replace outliers, keep all rows",
             "col.where(~mask, col.median())"),
            ("Row removal",
             f"[{COLOR_MEDIUM}]LOW (<1%)[/{COLOR_MEDIUM}]",
             "Only data-entry errors or instrument faults",
             "df.drop(index=outlier_indices)"),
            ("Isolation Forest",
             f"[{COLOR_HIGH}]HIGH / Multivariate[/{COLOR_HIGH}]",
             "Complex patterns invisible to univariate methods",
             "detector.detect_outliers_isolation_forest()"),
            ("Subgroup / separate model",
             f"[{COLOR_HIGH}]HIGH[/{COLOR_HIGH}]",
             "Outliers belong to a different regime (fraud, VIP)",
             "Train separate model on flagged subset"),
        ]:
            rec_table.add_row(strategy, sev, best_for, snippet)
        self.console.print(rec_table)

        # Per-column hints
        if summary is not None and not summary.empty:
            affected = summary[summary["Has_Outliers"]]
            if not affected.empty:
                hint_table = Table(
                    title="🔎 Column-Level Action Hints", box=ROUNDED,
                    title_style=f"bold {COLOR_HEADER}",
                    header_style=f"bold {COLOR_HEADER}",
                )
                hint_table.add_column("Column",   style="bold")
                hint_table.add_column("Outlier%", justify="center")
                hint_table.add_column("Skew",     justify="center")
                hint_table.add_column("Severity", justify="center")
                hint_table.add_column("Suggested Action", justify="left")

                for _, row in affected.iterrows():
                    pct   = row["Outlier_%"]
                    skew  = row["Skewness"]
                    color, label = _severity(pct)
                    skew_color = (COLOR_HIGH if abs(skew) > SKEW_HIGH
                                  else COLOR_MEDIUM if abs(skew) > SKEW_MODERATE
                                  else COLOR_GOOD)

                    # Smart action based on both outlier% AND skewness
                    if abs(skew) > SKEW_HIGH and pct < 2:
                        action = "⚠️ Likely false positives due to high skew — apply log-transform first, then re-run"
                    elif abs(skew) > SKEW_MODERATE and pct < 5:
                        action = "Verify manually — skewed distribution may inflate outlier count; consider log-transform"
                    elif pct > 10:
                        action = "Do NOT remove blindly — investigate; use robust model or subgroup analysis"
                    elif pct > 5:
                        action = "Winsorise to IQR bounds or log-transform; check for data-entry errors"
                    elif pct > 1:
                        action = "Cap or impute with median; add outlier indicator column if modeling"
                    else:
                        action = "Low impact — safe to cap, impute, or leave as-is for robust models"

                    hint_table.add_row(
                        row["Column"],
                        f"[{color}]{pct:.2f}%[/{color}]",
                        f"[{skew_color}]{skew:+.2f}[/{skew_color}]",
                        f"[{color}]{label}[/{color}]",
                        action,
                    )
                self.console.print(hint_table)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_distributions(
        self,
        columns: Optional[List[str]] = None,
        method:  str = "modified_z",
        max_cols: int = 6,
    ) -> None:
        """
        For each column: histogram + KDE + vertical lines for mean/median,
        plus a skewness and outlier count annotation.

        This is the primary tool to decide whether flagged outliers are
        genuine or artefacts of a skewed distribution.

        Parameters
        ----------
        columns  : Columns to plot (default = all numeric with outliers first).
        method   : Detection method used to annotate outlier count.
        max_cols : Maximum number of columns to plot.
        """
        if not self.numeric_columns:
            self.console.print(f"[{COLOR_MEDIUM}]No numeric columns to plot[/{COLOR_MEDIUM}]")
            return

        if columns:
            cols = [c for c in columns if c in self.numeric_columns]
        else:
            summary = self.get_outlier_summary(method)
            with_out = summary[summary["Has_Outliers"]]["Column"].tolist()
            clean    = summary[~summary["Has_Outliers"]]["Column"].tolist()
            cols = (with_out + clean)[:max_cols]

        n         = len(cols)
        ncols_g   = min(3, n)
        nrows_g   = (n + ncols_g - 1) // ncols_g

        fig, axes = plt.subplots(
            nrows_g, ncols_g,
            figsize=(6 * ncols_g, 4 * nrows_g),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        det_method = method if method in UNIVARIATE_METHODS else "modified_z"

        for i, col in enumerate(cols):
            ax     = axes_flat[i]
            series = self.data[col].dropna()
            dstat  = self._get_distribution_stats(col)
            res    = self._run_univariate(col, det_method)

            # Histogram + KDE
            ax.hist(series, bins=40, color="#AED6F1", alpha=0.7,
                    edgecolor="white", linewidth=0.4, density=True)
            try:
                kde_x = np.linspace(series.min(), series.max(), 300)
                kde   = scipy_stats.gaussian_kde(series)
                ax.plot(kde_x, kde(kde_x), color="#2E4057", linewidth=2, label="KDE")
            except Exception:
                pass

            # Mean and median lines
            ax.axvline(dstat["mean"],   color="#D72631", linewidth=1.5,
                       linestyle="--", label=f"Mean {dstat['mean']:.1f}")
            ax.axvline(dstat["median"], color="#27AE60", linewidth=1.5,
                       linestyle="-",  label=f"Median {dstat['median']:.1f}")

            # Skewness color
            skew_color = ("#D72631" if abs(dstat["skewness"]) > SKEW_HIGH
                          else "#F39C12" if abs(dstat["skewness"]) > SKEW_MODERATE
                          else "#27AE60")

            title_suffix = f"  [outliers: {res['outlier_count']:,}]" if res["outlier_count"] else ""
            ax.set_title(f"{col}{title_suffix}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Value", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.25)

            # Skewness annotation
            ax.annotate(
                f"skew={dstat['skewness']:+.2f}",
                xy=(0.03, 0.93), xycoords="axes fraction",
                fontsize=8, color=skew_color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

            # Range annotation
            ax.annotate(
                f"min={dstat['min']:.1f}  max={dstat['max']:.1f}",
                xy=(0.03, 0.83), xycoords="axes fraction",
                fontsize=7, color="#555555",
            )

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f"Distribution Analysis — {method.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def plot_outliers(
        self,
        column:   Optional[str] = None,
        method:   str = "modified_z",
        max_cols: int = 4,
    ) -> None:
        """
        Boxplot with flagged outlier points highlighted in red.

        Parameters
        ----------
        column   : Focus on a single column (optional).
        method   : Detection method.
        max_cols : Max columns when *column* is not given.
        """
        if not self.numeric_columns:
            self.console.print(f"[{COLOR_MEDIUM}]No numeric columns to plot[/{COLOR_MEDIUM}]")
            return

        if column:
            if column not in self.numeric_columns:
                self.console.print(f"[{COLOR_HIGH}]Column '{column}' not found or not numeric[/{COLOR_HIGH}]")
                return
            cols = [column]
        else:
            summary = self.get_outlier_summary(method)
            with_out = summary[summary["Outlier_Count"] > 0]["Column"].tolist()
            cols = (with_out or self.numeric_columns)[:max_cols]

        n         = len(cols)
        ncols_g   = min(2, n)
        nrows_g   = (n + 1) // 2

        fig, axes = plt.subplots(
            nrows_g, ncols_g,
            figsize=(6 * ncols_g, 4 * nrows_g),
            squeeze=False,
        )
        axes_flat = axes.flatten()
        det_method = method if method in UNIVARIATE_METHODS else "modified_z"

        for i, col in enumerate(cols):
            ax     = axes_flat[i]
            series = self.data[col].dropna()

            bp = ax.boxplot(series, vert=True, patch_artist=True)
            bp["boxes"][0].set_facecolor("#AED6F1")
            bp["boxes"][0].set_alpha(0.8)
            ax.set_title(f"Boxplot: {col}", fontsize=11, fontweight="bold")
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)

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

    # ------------------------------------------------------------------
    # Handling
    # ------------------------------------------------------------------

    def remove_outliers(
        self,
        method:  str = "modified_z",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Drop rows that are outliers in any of the given columns."""
        cols = [c for c in (columns or self.numeric_columns)
                if c in self.numeric_columns]
        bad: set = set()
        for col in cols:
            bad.update(self._run_univariate(col, method)["indices"])
        if bad:
            self.console.print(
                f"[{COLOR_GOOD}]✅ Removed {len(bad):,} outlier rows "
                f"({len(bad)/len(self.data)*100:.2f}% of dataset)[/{COLOR_GOOD}]"
            )
            return self.data.drop(index=list(bad)).reset_index(drop=True)
        self.console.print(f"[{COLOR_MEDIUM}]No outliers found to remove[/{COLOR_MEDIUM}]")
        return self.data.copy()

    def cap_outliers(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Winsorise outliers to IQR bounds. Returns new DataFrame."""
        cols   = [c for c in (columns or self.numeric_columns) if c in self.numeric_columns]
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
        method:  str = "modified_z",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Replace detected outliers with column median. Returns new DataFrame."""
        cols    = [c for c in (columns or self.numeric_columns) if c in self.numeric_columns]
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
        method:        str = "modified_z",
        columns:       Optional[List[str]] = None,
        sample_size:   int = 20,
        random_sample: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Display outlier rows per column with full context."""
        if method not in UNIVARIATE_METHODS:
            raise ValueError(f"method must be one of {sorted(UNIVARIATE_METHODS)}")
        if sample_size <= 0:
            raise ValueError("sample_size must be > 0")

        cols    = [c for c in (columns or self.numeric_columns) if c in self.numeric_columns]
        results: Dict[str, pd.DataFrame] = {}

        with pd.option_context("display.max_columns", None, "display.max_rows", None,
                               "display.width", None, "display.max_colwidth", 50):
            for col in cols:
                res = self._run_univariate(col, method)
                if res["outlier_count"] == 0:
                    continue

                outlier_rows = self.data.reindex(res["indices"]).copy()
                if "Outlier_Value" in outlier_rows.columns:
                    outlier_rows.drop(columns=["Outlier_Value"], inplace=True)
                outlier_rows.insert(0, "Outlier_Value", outlier_rows[col])

                total = len(outlier_rows)
                display_df = (
                    outlier_rows.sample(n=sample_size, random_state=42)
                    if total > sample_size and random_sample
                    else outlier_rows.head(sample_size) if total > sample_size
                    else outlier_rows
                )
                results[col] = outlier_rows

                _, severity = _severity(res["percentage"])
                dstat       = self._get_distribution_stats(col)
                sep = "─" * 90
                print(f"\n{sep}")
                print(
                    f"📌 Column      : {col}\n"
                    f"📊 Outliers    : {total:,}  ({res['percentage']:.2f}%)\n"
                    f"🎯 Severity    : {severity}\n"
                    f"📐 Skewness    : {dstat['skewness']:+.2f}  {dstat['skew_label']}\n"
                    f"📏 Range       : {dstat['min']:.2f} → {dstat['max']:.2f}\n"
                    + (f"⚠️  {dstat['skew_warning']}\n" if dstat["skew_warning"] else "") +
                    f"👀 Displaying  : {len(display_df)} of {total} rows"
                )
                print(sep)
                print(display_df.to_string())
                print()

        if not results:
            print("\n✅ No outliers detected in any column.\n")

        return results