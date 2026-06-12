# missing_values_analyzer.py
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.box import ROUNDED

class MissingValuesAnalyzer:
    """
    Class to analyze and report missing values in a dataset.
    Uses a cohesive amber/orange-based color palette to represent
    severity levels of missing data (low -> medium -> high).
    """

    def __init__(self, data: pd.DataFrame):
        """
        Constructor for MissingValuesAnalyzer class.

        Args:
            data: Input DataFrame to analyze
        """
        self.data = data
        self.console = Console()

        # Cohesive color theme (warm amber/orange family for "missingness")
        self.color_header = "#2E4057"      # Deep slate blue (headers/titles)
        self.color_high = "#D72631"        # Strong red (high severity, >20%)
        self.color_medium = "#F39C12"      # Amber/orange (medium severity, 5-20%)
        self.color_low = "#F7DC6F"         # Soft yellow (low severity, <5%)
        self.color_good = "#27AE60"        # Green (no issues / good status)

        # Threshold for flagging rows as candidates for removal
        self.row_missing_threshold = 3

    def get_missing_summary(self) -> pd.DataFrame:
        """
        Get comprehensive summary of missing values for each column.

        Returns:
            DataFrame containing missing values summary for each column
        """
        missing_summary = []

        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            missing_pct = (missing_count / len(self.data)) * 100
            dtype_category = self._get_dtype_category(self.data[col])

            missing_summary.append({
                'Column': col,
                'Data_Type': dtype_category,
                'Missing_Count': missing_count,
                'Missing_Percentage': missing_pct,
                'Non_Missing_Count': len(self.data) - missing_count
            })

        summary_df = pd.DataFrame(missing_summary)
        summary_df = summary_df.sort_values('Missing_Percentage', ascending=False)
        return summary_df

    def _get_dtype_category(self, column: pd.Series) -> str:
        """
        Categorize column data type.

        Args:
            column: Pandas Series to categorize

        Returns:
            String representing data type category
        """
        if pd.api.types.is_numeric_dtype(column):
            return 'Numeric'
        elif pd.api.types.is_datetime64_any_dtype(column):
            return 'DateTime'
        elif pd.api.types.is_categorical_dtype(column):
            return 'Categorical'
        elif pd.api.types.is_bool_dtype(column):
            return 'Boolean'
        else:
            return 'Object/String'

    def get_columns_with_missing(self) -> list:
        """
        Get list of columns that have at least one missing value.

        Returns:
            List of column names with missing values
        """
        return [col for col in self.data.columns if self.data[col].isnull().any()]

    def get_total_missing_count(self) -> int:
        """
        Calculate total number of missing values in entire dataset.

        Returns:
            Total count of missing values across all columns
        """
        return int(self.data.isnull().sum().sum())

    def get_total_missing_percentage(self) -> float:
        """
        Calculate percentage of missing values in entire dataset.

        Returns:
            Percentage of missing values relative to total cells
        """
        total_cells = self.data.shape[0] * self.data.shape[1]
        total_missing = self.get_total_missing_count()
        if total_cells == 0:
            return 0.0
        return (total_missing / total_cells) * 100

    def get_missing_by_row(self) -> pd.DataFrame:
        """
        Analyze missing values by row.

        Returns:
            DataFrame showing missing count per row
        """
        missing_per_row = self.data.isnull().sum(axis=1)
        rows_with_missing = missing_per_row[missing_per_row > 0]

        summary_by_row = pd.DataFrame({
            'Row_Index': rows_with_missing.index,
            'Missing_Count': rows_with_missing.values,
            'Missing_Percentage': (rows_with_missing.values / self.data.shape[1]) * 100
        })
        return summary_by_row.sort_values('Missing_Count', ascending=False)

    def get_rows_above_threshold(self, threshold: int = None) -> pd.DataFrame:
        """
        Get rows whose missing-value count is >= threshold.
        These are candidate rows for removal.

        Args:
            threshold: Minimum number of missing values to flag a row.
                       Defaults to self.row_missing_threshold (3).

        Returns:
            DataFrame of flagged rows, sorted by missing count (desc).
        """
        if threshold is None:
            threshold = self.row_missing_threshold

        missing_by_row = self.get_missing_by_row()
        return missing_by_row[missing_by_row['Missing_Count'] >= threshold]

    def display_detailed_report(self):
        """
        Display detailed report of missing values using Rich library.
        """
        self.console.print(f"\n[bold {self.color_header}]📊 MISSING VALUES ANALYSIS REPORT[/bold {self.color_header}]")
        self.console.print(f"[dim]{'='*60}[/dim]\n")

        summary_df = self.get_missing_summary()
        columns_with_missing = self.get_columns_with_missing()
        total_missing = self.get_total_missing_count()
        total_missing_pct = self.get_total_missing_percentage()

        self._display_overall_stats(columns_with_missing, total_missing, total_missing_pct)

        if len(columns_with_missing) > 0:
            self._display_column_missing_table(summary_df)
            self._display_recommendations(summary_df)
        else:
            self.console.print(f"\n[{self.color_good}]✓ Excellent! No missing values found in the dataset![/{self.color_good}]")

        if total_missing > 0:
            self._display_row_missing_info()

    def _display_overall_stats(self, columns_with_missing: list, total_missing: int, total_missing_pct: float):
        """
        Display overall missing values statistics.
        """
        stats_table = Table(
            title="📈 Overall Statistics",
            title_style=f"bold {self.color_header}",
            box=ROUNDED,
            header_style=f"bold {self.color_header}"
        )
        stats_table.add_column("Metric", justify="left", style="bold")
        stats_table.add_column("Value", justify="right")

        total_cells = self.data.shape[0] * self.data.shape[1]
        stats_table.add_row("Total Cells in Dataset", f"{total_cells:,}")
        stats_table.add_row("Columns with Missing Values", f"{len(columns_with_missing)} out of {self.data.shape[1]}")
        stats_table.add_row("Total Missing Values", f"{total_missing:,}")
        stats_table.add_row("Total Missing Percentage", f"{total_missing_pct:.2f}%")

        self.console.print(stats_table)

    def _display_column_missing_table(self, summary_df: pd.DataFrame):
        """
        Display detailed table of missing values by column.
        """
        column_table = Table(
            title="🔍 Missing Values by Column",
            title_style=f"bold {self.color_header}",
            box=ROUNDED,
            header_style=f"bold {self.color_header}"
        )
        column_table.add_column("Column Name", justify="left", no_wrap=True)
        column_table.add_column("Data Type", justify="center")
        column_table.add_column("Missing Count", justify="center")
        column_table.add_column("Missing %", justify="center")
        column_table.add_column("Non-Missing", justify="center")

        columns_with_missing = summary_df[summary_df['Missing_Count'] > 0]

        for _, row in columns_with_missing.iterrows():
            missing_pct = row['Missing_Percentage']
            if missing_pct > 20:
                color = self.color_high
            elif missing_pct > 5:
                color = self.color_medium
            else:
                color = self.color_low

            column_table.add_row(
                f"[{color}]{row['Column']}[/{color}]",
                row['Data_Type'],
                f"[{color}]{int(row['Missing_Count']):,}[/{color}]",
                f"[{color}]{row['Missing_Percentage']:.2f}%[/{color}]",
                f"{int(row['Non_Missing_Count']):,}"
            )
        self.console.print(column_table)

    def _display_recommendations(self, summary_df: pd.DataFrame):
        """
        Display professional, statistically-grounded recommendations
        for handling missing values, tailored to severity and data type.
        """
        recommendations_table = Table(
            title="💡 Recommendations",
            title_style=f"bold {self.color_header}",
            box=ROUNDED,
            header_style=f"bold {self.color_header}"
        )
        recommendations_table.add_column("Column", justify="left")
        recommendations_table.add_column("Missing %", justify="center")
        recommendations_table.add_column("Severity", justify="center")
        recommendations_table.add_column("Recommended Action", justify="left")

        for _, row in summary_df.iterrows():
            if row['Missing_Percentage'] > 0:
                missing_pct = row['Missing_Percentage']
                dtype = row['Data_Type']

                if missing_pct > 50:
                    severity = f"[{self.color_high}]Critical[/{self.color_high}]"
                    action = ("Drop column or treat as MNAR; if domain-critical, "
                              "consider a 'missing' indicator flag and model-based imputation (e.g., MICE)")
                elif missing_pct > 20:
                    severity = f"[{self.color_high}]High[/{self.color_high}]"
                    if dtype == 'Numeric':
                        action = ("Use multiple imputation (MICE) or KNN imputation; "
                                  "add a missingness indicator column before modeling")
                    else:
                        action = ("Impute with mode, a dedicated 'Unknown' category, "
                                  "or model-based imputation; add a missingness indicator")
                elif missing_pct > 5:
                    severity = f"[{self.color_medium}]Moderate[/{self.color_medium}]"
                    if dtype == 'Numeric':
                        action = "Impute with median (robust to outliers) or use regression imputation"
                    elif dtype in ('DateTime',):
                        action = "Impute via forward/backward fill or interpolation based on time order"
                    else:
                        action = "Impute with mode or most frequent category; review for systematic patterns"
                else:
                    severity = f"[{self.color_low}]Low[/{self.color_low}]"
                    action = "Low impact: impute with mean/median/mode or drop affected rows safely"

                recommendations_table.add_row(
                    row['Column'],
                    f"{row['Missing_Percentage']:.1f}%",
                    severity,
                    action
                )
        self.console.print(recommendations_table)

    def _display_row_missing_info(self):
        """
        Display information about rows with missing values, and flag
        rows that meet/exceed the row_missing_threshold (default: 3)
        as candidates for removal.
        """
        missing_by_row = self.get_missing_by_row()
        flagged_rows = self.get_rows_above_threshold()

        if len(missing_by_row) > 0:
            row_table = Table(
                title="📊 Rows with Missing Values",
                title_style=f"bold {self.color_header}",
                box=ROUNDED,
                header_style=f"bold {self.color_header}"
            )
            row_table.add_column("Row Index", justify="center")
            row_table.add_column("Missing Count", justify="center")
            row_table.add_column("Missing Percentage", justify="center")
            row_table.add_column("Suggested Action", justify="left")

            for _, row in missing_by_row.head(10).iterrows():
                if row['Missing_Count'] >= self.row_missing_threshold:
                    rec = f"⚠️ ≥{self.row_missing_threshold} missing — candidate for removal"
                    color = self.color_high
                elif row['Missing_Percentage'] > 20:
                    rec = "Impute values"
                    color = self.color_medium
                else:
                    rec = "Impute or keep"
                    color = self.color_low

                row_table.add_row(
                    str(row['Row_Index']),
                    f"[{color}]{int(row['Missing_Count'])}[/{color}]",
                    f"{row['Missing_Percentage']:.1f}%",
                    f"[{color}]{rec}[/{color}]"
                )

            if len(missing_by_row) > 10:
                row_table.caption = f"Showing top 10 rows out of {len(missing_by_row)} rows with missing values"
            self.console.print(row_table)

        # Summary block: count of rows that meet/exceed threshold
        n_flagged = len(flagged_rows)
        if n_flagged > 0:
            pct_flagged = (n_flagged / len(self.data)) * 100
            self.console.print(
                f"\n[bold {self.color_high}]🗑️  {n_flagged:,} row(s) ({pct_flagged:.2f}% of dataset) "
                f"have {self.row_missing_threshold}+ missing values.[/bold {self.color_high}]"
            )
            self.console.print(
                f"[{self.color_medium}]   Decide whether to drop these rows: "
                f"`df.drop(index=analyzer.get_rows_above_threshold()['Row_Index'])`[/{self.color_medium}]"
            )
        else:
            self.console.print(
                f"\n[{self.color_good}]✓ No rows have {self.row_missing_threshold}+ missing values "
                f"(no removal candidates by row-count rule).[/{self.color_good}]"
            )

    def export_report_to_csv(self, filename: str = "missing_values_report.csv"):
        """
        Export missing values report to CSV file.
        """
        summary_df = self.get_missing_summary()
        summary_df.to_csv(filename, index=False)
        self.console.print(f"\n[{self.color_good}]✓ Report exported to '{filename}'[/{self.color_good}]")

    def get_simple_text_report(self) -> str:
        """
        Get a simple text report of missing values.

        Returns:
            String containing simple text report
        """
        columns_with_missing = self.get_columns_with_missing()
        total_missing = self.get_total_missing_count()
        total_missing_pct = self.get_total_missing_percentage()

        report = []
        report.append("=" * 60)
        report.append("MISSING VALUES ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal Missing Values: {total_missing:,}")
        report.append(f"Total Missing Percentage: {total_missing_pct:.2f}%")
        report.append(f"Columns with Missing Values: {len(columns_with_missing)} out of {self.data.shape[1]}")

        if columns_with_missing:
            report.append("\nColumns with missing values:")
            summary_df = self.get_missing_summary()
            for _, row in summary_df[summary_df['Missing_Count'] > 0].iterrows():
                report.append(f"  - {row['Column']}: {int(row['Missing_Count']):,} missing ({row['Missing_Percentage']:.2f}%)")
        else:
            report.append("\n✓ No missing values found!")

        flagged_rows = self.get_rows_above_threshold()
        if len(flagged_rows) > 0:
            pct_flagged = (len(flagged_rows) / len(self.data)) * 100
            report.append(f"\nRows with {self.row_missing_threshold}+ missing values: "
                           f"{len(flagged_rows):,} ({pct_flagged:.2f}% of dataset) — review for removal")

        report.append("\n" + "=" * 60)
        return "\n".join(report)