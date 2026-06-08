 # missing_values_analyzer.py
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.box import ROUNDED

class MissingValuesAnalyzer:
    """
    Class to analyze and report missing values in a dataset.
    Colors are neutral and natural (grey, beige, etc.).
    """

    def __init__(self, data: pd.DataFrame):
        """
        Constructor for MissingValuesAnalyzer class.

        Args:
            data: Input DataFrame to analyze
        """
        self.data = data
        self.console = Console()

        # Neutral and natural color theme
        self.color_header = "#4F4F4F"      # Dark Gray
        self.color_high = "#8B0000"        # DarkRed (for high missing)
        self.color_medium = "#A0522D"      # Sienna (for medium missing)
        self.color_low = "#D3D3D3"         # LightGray (for low missing)

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
            self.console.print(f"\n[green]✓ Excellent! No missing values found in the dataset![/green]")

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
                pct_color = "red"
            elif missing_pct > 5:
                color = self.color_medium
                pct_color = "yellow"
            else:
                color = self.color_low
                pct_color = "white"

            column_table.add_row(
                f"[{color}]{row['Column']}[/{color}]",
                row['Data_Type'],
                f"[{color}]{int(row['Missing_Count']):,}[/{color}]",
                f"[{pct_color}]{row['Missing_Percentage']:.2f}%[/{pct_color}]",
                f"{int(row['Non_Missing_Count']):,}"
            )
        self.console.print(column_table)

    def _display_recommendations(self, summary_df: pd.DataFrame):
        """
        Display recommendations for handling missing values.
        """
        recommendations_table = Table(
            title="💡 Recommendations",
            title_style=f"bold {self.color_header}",
            box=ROUNDED,
            header_style=f"bold {self.color_header}"
        )
        recommendations_table.add_column("Column", justify="left")
        recommendations_table.add_column("Missing %", justify="center")
        recommendations_table.add_column("Recommended Action", justify="left")

        for _, row in summary_df.iterrows():
            if row['Missing_Percentage'] > 0:
                missing_pct = row['Missing_Percentage']
                if missing_pct > 50:
                    action = "❌ Consider dropping this column"
                elif missing_pct > 20:
                    action = "⚠️ Consider dropping or imputing with advanced methods"
                elif missing_pct > 5:
                    if row['Data_Type'] == 'Numeric':
                        action = "📊 Impute with mean/median"
                    else:
                        action = "🏷️ Impute with mode or 'Unknown'"
                else:
                    action = "✓ Impute or drop rows (low impact)"

                recommendations_table.add_row(
                    row['Column'],
                    f"{row['Missing_Percentage']:.1f}%",
                    action
                )
        self.console.print(recommendations_table)

    def _display_row_missing_info(self):
        """
        Display information about rows with missing values.
        """
        missing_by_row = self.get_missing_by_row()
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
            row_table.add_column("Recommendation", justify="left")

            for _, row in missing_by_row.head(10).iterrows():
                if row['Missing_Percentage'] > 50:
                    rec = "Consider dropping this row"
                    color = "red"
                elif row['Missing_Percentage'] > 20:
                    rec = "Impute values"
                    color = "yellow"
                else:
                    rec = "Impute or keep"
                    color = "white"

                row_table.add_row(
                    str(row['Row_Index']),
                    f"[{color}]{int(row['Missing_Count'])}[/{color}]",
                    f"{row['Missing_Percentage']:.1f}%",
                    rec
                )

            if len(missing_by_row) > 10:
                row_table.caption = f"Showing top 10 rows out of {len(missing_by_row)} rows with missing values"
            self.console.print(row_table)

    def export_report_to_csv(self, filename: str = "missing_values_report.csv"):
        """
        Export missing values report to CSV file.
        """
        summary_df = self.get_missing_summary()
        summary_df.to_csv(filename, index=False)
        self.console.print(f"\n[green]✓ Report exported to '{filename}'[/green]")

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

        report.append("\n" + "=" * 60)
        return "\n".join(report)