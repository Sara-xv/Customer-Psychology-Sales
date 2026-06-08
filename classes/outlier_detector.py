# classes/outlier_detector.py
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.box import ROUNDED
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest


class OutlierDetector:
    """
    Advanced Outlier Detection Class
    Supports: IQR, Z-Score, Modified Z-Score, Isolation Forest
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.console = Console()
        
        # Color theme
        self.color_header = "#2E8B57"   # SeaGreen
        self.color_high = "#B22222"     # FireBrick
        self.color_medium = "#D2691E"
        self.color_low = "#4682B4"

    # ====================== تشخیص outlier ======================
    
    def detect_outliers_iqr(self, column: str) -> dict:
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            return {'method': 'IQR', 'outlier_count': 0, 'indices': []}
        
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = self.data[(self.data[column] < lower) | (self.data[column] > upper)]
        return {
            'method': 'IQR',
            'outlier_count': len(outliers),
            'percentage': len(outliers)/len(self.data)*100,
            'lower_bound': lower,
            'upper_bound': upper,
            'indices': outliers.index.tolist()
        }

    def detect_outliers_zscore(self, column: str, threshold: float = 3.0) -> dict:
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            return {'method': 'Z-Score', 'outlier_count': 0, 'indices': []}
        
        mean = self.data[column].mean()
        std = self.data[column].std()
        z_scores = np.abs((self.data[column] - mean) / std)
        outliers = self.data[z_scores > threshold]
        
        return {
            'method': 'Z-Score',
            'outlier_count': len(outliers),
            'percentage': len(outliers)/len(self.data)*100,
            'threshold': threshold,
            'indices': outliers.index.tolist()
        }

    def detect_outliers_modified_zscore(self, column: str, threshold: float = 3.5) -> dict:
        """Modified Z-Score - Robust method using Median and MAD"""
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            return {'method': 'Modified Z-Score', 'outlier_count': 0, 'indices': []}
        
        median = self.data[column].median()
        mad = np.median(np.abs(self.data[column] - median))
        if mad == 0:
            return {'method': 'Modified Z-Score', 'outlier_count': 0, 'indices': []}
        
        modified_z = np.abs(0.6745 * (self.data[column] - median) / mad)
        outliers = self.data[modified_z > threshold]
        
        return {
            'method': 'Modified Z-Score',
            'outlier_count': len(outliers),
            'percentage': len(outliers)/len(self.data)*100,
            'threshold': threshold,
            'indices': outliers.index.tolist()
        }

    def detect_outliers_isolation_forest(self, columns: list = None, contamination: float = 0.05) -> dict:
        """Isolation Forest - Best for multivariate outliers"""
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        numeric_df = self.data[columns].dropna()
        if len(numeric_df) == 0:
            return {'method': 'Isolation Forest', 'outlier_count': 0, 'indices': []}
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        preds = iso_forest.fit_predict(numeric_df)
        outliers = numeric_df[preds == -1]
        
        return {
            'method': 'Isolation Forest',
            'outlier_count': len(outliers),
            'percentage': len(outliers)/len(self.data)*100,
            'indices': outliers.index.tolist(),
            'contamination': contamination
        }

    # ====================== خلاصه گزارش ======================
    def get_outlier_summary(self, method: str = 'iqr') -> pd.DataFrame:
        summary = []
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                res = self.detect_outliers_iqr(col)
            elif method == 'zscore':
                res = self.detect_outliers_zscore(col)
            elif method == 'modified_z':
                res = self.detect_outliers_modified_zscore(col)
            else:
                continue
                
            summary.append({
                'Column': col,
                'Method': res['method'],
                'Outlier_Count': res['outlier_count'],
                'Outlier_%': round(res['percentage'], 2)
            })
        
        return pd.DataFrame(summary).sort_values('Outlier_%', ascending=False)

    # ====================== نمایش گزارش ======================
    def display_detailed_report(self, method: str = 'modified_z'):
        self.console.print(f"\n[bold {self.color_header}]📊 ADVANCED OUTLIER DETECTION REPORT[/bold {self.color_header}]")
        self.console.print("="*70)
        
        if method == 'isolation':
            result = self.detect_outliers_isolation_forest()
            self.console.print(f"[bold]Method:[/] Isolation Forest (Multivariate)")
        else:
            summary = self.get_outlier_summary(method)
            total_outliers = summary['Outlier_Count'].sum()
            
            # جدول کلی
            stats = Table(title="Overall Statistics", box=ROUNDED)
            stats.add_column("Metric")
            stats.add_column("Value", justify="right")
            stats.add_row("Numeric Columns", str(len(summary)))
            stats.add_row("Total Outliers", str(total_outliers))
            stats.add_row("Method", method.replace('_', ' ').title())
            self.console.print(stats)
            
            # جدول جزئیات
            self._display_column_table(summary)

        self._display_recommendations()
        self.plot_outliers(method=method)

    def _display_column_table(self, summary: pd.DataFrame):
        table = Table(title="Outliers by Column", box=ROUNDED, title_style="bold")
        table.add_column("Column")
        table.add_column("Outlier Count")
        table.add_column("Outlier %")

        for _, row in summary[summary['Outlier_Count'] > 0].iterrows():
            pct = row['Outlier_%']
            color = self.color_high if pct > 5 else self.color_medium if pct > 1 else self.color_low
            table.add_row(
                row['Column'],
                f"[{color}]{int(row['Outlier_Count'])}[/{color}]",
                f"[{color}]{pct}%[/{color}]"
            )
        self.console.print(table)

    def _display_recommendations(self):
        rec_table = Table(title="💡 Recommendations", box=ROUNDED)
        rec_table.add_column("Action")
        rec_table.add_column("When to Use")
        rec_table.add_row("Capping / Winsorizing", "Outliers moderate (1-10%)")
        rec_table.add_row("Remove rows", "Very few outliers & small dataset")
        rec_table.add_row("Use Robust Models", "Many outliers or high dimensions")
        rec_table.add_row("Transform variable", "Skewed distribution (log, sqrt)")
        self.console.print(rec_table)

    # ====================== نمودار ======================
    def plot_outliers(self, column: str = None, method: str = 'modified_z'):
        """رسم نمودارهای تشخیص outlier"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if column and column in numeric_cols:
            cols_to_plot = [column]
        else:
            # فقط ۴ ستون با بیشترین outlier
            summary = self.get_outlier_summary(method)
            cols_to_plot = summary[summary['Outlier_Count'] > 0].head(4)['Column'].tolist()
            if not cols_to_plot:
                cols_to_plot = numeric_cols[:4]

        if not cols_to_plot:
            return

        plt.figure(figsize=(15, 10))
        for i, col in enumerate(cols_to_plot, 1):
            plt.subplot(2, 2, i)
            sns.boxplot(y=self.data[col])
            plt.title(f'Box Plot - {col}')
            
            # اضافه کردن نقاط outlier
            if method == 'modified_z':
                res = self.detect_outliers_modified_zscore(col)
            elif method == 'iqr':
                res = self.detect_outliers_iqr(col)
            else:
                res = self.detect_outliers_zscore(col)
                
            if res['outlier_count'] > 0:
                outliers = self.data.loc[res['indices'], col]
                plt.scatter([0]*len(outliers), outliers, color='red', alpha=0.7, label=f'Outliers ({len(outliers)})')
                plt.legend()

        plt.tight_layout()
        plt.show()
        
        # Histogram + KDE
        if len(cols_to_plot) > 0:
            plt.figure(figsize=(12, 6))
            for col in cols_to_plot:
                sns.histplot(self.data[col], kde=True, label=col, alpha=0.6)
            plt.title("Distribution of Selected Columns")
            plt.legend()
            plt.show()