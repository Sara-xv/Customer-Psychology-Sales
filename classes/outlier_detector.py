import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.box import ROUNDED
import matplotlib.pyplot as plt


class OutlierDetector:
    """
    Class to detect and analyze outliers in a dataset.
    Supports IQR and Z-Score methods.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Constructor for OutlierDetector class.

        Args:
            data: Input DataFrame to analyze
        """
        self.data = data.copy()  # جلوگیری از تغییر دیتاست اصلی
        self.console = Console()
        
        # Color theme
        self.color_header = "#4F4F4F"
        self.color_high = "#8B0000"      # Dark Red
        self.color_medium = "#D2691E"    # Chocolate
        self.color_low = "#4682B4"       # Steel Blue

    def detect_outliers_iqr(self, column: str) -> dict:
        """Detect outliers using IQR method"""
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            return {'outlier_count': 0, 'outlier_indices': [], 'lower_bound': None, 'upper_bound': None}
        
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(self.data)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_indices': outliers.index.tolist()
        }

    def detect_outliers_zscore(self, column: str, threshold: float = 3.0) -> dict:
        """Detect outliers using Z-Score method"""
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            return {'outlier_count': 0, 'outlier_percentage': 0, 'outlier_indices': []}
        
        mean = self.data[column].mean()
        std = self.data[column].std()
        
        z_scores = np.abs((self.data[column] - mean) / std)
        outliers = self.data[z_scores > threshold]
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(self.data)) * 100,
            'threshold': threshold,
            'outlier_indices': outliers.index.tolist()
        }

    def get_outlier_summary(self, method: str = 'iqr') -> pd.DataFrame:
        """Get summary of outliers for all numeric columns"""
        summary = []
        
        for col in self.data.select_dtypes(include=[np.number]).columns:
            if method.lower() == 'iqr':
                result = self.detect_outliers_iqr(col)
            else:
                result = self.detect_outliers_zscore(col)
            
            summary.append({
                'Column': col,
                'Outlier_Count': result['outlier_count'],
                'Outlier_Percentage': result['outlier_percentage'],
                'Method': method.upper()
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('Outlier_Percentage', ascending=False)
        return summary_df

    def display_detailed_report(self, method: str = 'iqr'):
        """Display beautiful outlier analysis report"""
        self.console.print(f"\n[bold {self.color_header}]📊 OUTLIER DETECTION REPORT[/bold {self.color_header}]")
        self.console.print(f"[dim]{'='*60}[/dim]\n")

        summary_df = self.get_outlier_summary(method)
        total_outliers = summary_df['Outlier_Count'].sum()

        # Overall Stats
        stats_table = Table(title="📈 Overall Statistics", box=ROUNDED, title_style=f"bold {self.color_header}")
        stats_table.add_column("Metric", style="bold")
        stats_table.add_column("Value", justify="right")
        
        stats_table.add_row("Total Numeric Columns", str(len(summary_df)))
        stats_table.add_row("Total Outliers Detected", f"{total_outliers:,}")
        stats_table.add_row("Method Used", method.upper())
        self.console.print(stats_table)

        # Detailed Table
        if total_outliers > 0:
            detail_table = Table(title=f"🔍 Outliers by Column ({method.upper()})", 
                               box=ROUNDED, title_style=f"bold {self.color_header}")
            detail_table.add_column("Column", justify="left")
            detail_table.add_column("Outlier Count", justify="center")
            detail_table.add_column("Outlier %", justify="center")

            for _, row in summary_df[summary_df['Outlier_Count'] > 0].iterrows():
                pct = row['Outlier_Percentage']
                if pct > 5:
                    color = self.color_high
                elif pct > 1:
                    color = self.color_medium
                else:
                    color = self.color_low
                
                detail_table.add_row(
                    row['Column'],
                    f"[{color}]{int(row['Outlier_Count'])}[/{color}]",
                    f"[{color}]{pct:.2f}%[/{color}]"
                )
            self.console.print(detail_table)
            
            self._display_recommendations(summary_df)
        else:
            self.console.print("[green]✓ Excellent! No outliers detected in the dataset.[/green]")

    def _display_recommendations(self, summary_df: pd.DataFrame):
        """Show recommendations for handling outliers"""
        rec_table = Table(title="💡 Recommendations", box=ROUNDED, title_style=f"bold {self.color_header}")
        rec_table.add_column("Column", justify="left")
        rec_table.add_column("Outlier %", justify="center")
        rec_table.add_column("Suggested Action", justify="left")

        for _, row in summary_df[summary_df['Outlier_Count'] > 0].iterrows():
            pct = row['Outlier_Percentage']
            if pct > 10:
                action = "⚠️ Consider removing or capping (high impact)"
            elif pct > 3:
                action = "📊 Winsorize or use Robust Scaler"
            else:
                action = "✓ Keep or use mild transformation"
            
            rec_table.add_row(row['Column'], f"{pct:.2f}%", action)
        
        self.console.print(rec_table)

    def get_outlier_indices(self, column: str, method: str = 'iqr') -> list:
        """Return indices of outliers for a specific column"""
        if method.lower() == 'iqr':
            return self.detect_outliers_iqr(column)['outlier_indices']
        else:
            return self.detect_outliers_zscore(column)['outlier_indices']

    def export_report_to_csv(self, filename: str = "outliers_report.csv"):
        """Export outlier summary to CSV"""
        summary_df = self.get_outlier_summary()
        summary_df.to_csv(filename, index=False)
        self.console.print(f"\n[green]✓ Outlier report exported to '{filename}'[/green]")