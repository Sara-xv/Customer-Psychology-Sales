import pickle
from pathlib import Path
 
import numpy as np
import pandas as pd
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.terminal_theme import TerminalTheme
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import classification_report, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
 
# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
 
# Only psychological features are used as model input.
# Behavioral columns (total_spent, total_orders, ...) are NOT used as features
# because in a real app we only know the customer's psychological profile at
# acquisition time — not their purchase history.
PSYCH_FEATURES = [
    "stress_level", "anxiety_score", "self_esteem", "impulsiveness",
    "optimism_score", "life_satisfaction", "social_media_dependency",
]
 
SEGMENT_COLORS = {
    "Champions":           "#4CAF50",
    "Loyal Customers":     "#8BC34A",
    "Big Spenders":        "#42A5F5",
    "Potential Loyalists": "#FFB300",
    "Regular Customers":   "#9E9E9E",
    "At Risk / Low Value": "#C44536",
}
 
HTML_THEME = TerminalTheme(
    (18, 18, 18), (230, 230, 230),
    [(0,0,0),(205,49,49),(13,188,121),(229,229,16),
     (36,114,200),(188,63,188),(17,168,205),(229,229,229)],
    [(102,102,102),(241,76,76),(35,209,139),(245,245,67),
     (59,142,234),(214,112,214),(41,184,219),(255,255,255)],
)
 
ACCENT  = "#A784B6"
HEADER  = "#842958"
WARNING = "#E0A458"
ERROR   = "#C44536"
OK      = "#4CAF50"
VALUE   = "#E8E8E8"
BORDER  = "#5A5A5A"
 
# ─────────────────────────────────────────────────────────────────────────────
# RFM HELPERS (inline — no external dependency)
# ─────────────────────────────────────────────────────────────────────────────
 
def _label_segment(f: int, m: int) -> str:
    if f >= 4 and m >= 4: return "Champions"
    if f >= 4 and m <= 3: return "Loyal Customers"
    if f <= 2 and m >= 4: return "Big Spenders"
    if f == 3 and m == 3: return "Potential Loyalists"
    if f <= 2 and m <= 2: return "At Risk / Low Value"
    return "Regular Customers"
 
 
def _build_rfm_segments(data: pd.DataFrame) -> pd.DataFrame:
    """Add F_Score, M_Score, Customer_Segment to a copy of data."""
    data = data.copy()
    data["F_Score"] = pd.qcut(
        data["total_orders"].rank(method="first"), 5, labels=range(1, 6),
    ).astype(int)
    data["M_Score"] = pd.qcut(
        data["total_spent"].rank(method="first"), 5, labels=range(1, 6),
    ).astype(int)
    data["Customer_Segment"] = [
        _label_segment(f, m) for f, m in zip(data["F_Score"], data["M_Score"])
    ]
    return data
 
 
# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION RULES
# ─────────────────────────────────────────────────────────────────────────────
 
# Each rule: (feature, direction, threshold, tag, recommendation text)
# "high" → triggers when feature > threshold
# "low"  → triggers when feature < threshold
RULES: list[tuple[str, str, float, str, str]] = [
    ("stress_level", "high", 6.5, "Stress Management",
     "High stress detected. Use simple, low-friction promotions (direct discount codes) "
     "rather than multi-step campaigns. Reduce cognitive load in all touchpoints."),
 
    ("anxiety_score", "high", 6.5, "Trust Building",
     "High anxiety detected. Emphasize money-back guarantees, social proof (reviews), "
     "and easy-access support. Reduce perceived purchase risk."),
 
    ("impulsiveness", "high", 6.5, "Flash Offers",
     "High impulsiveness detected. Deploy limited-time flash sales and countdown timers. "
     "This customer responds strongly to urgency-based triggers."),
 
    ("impulsiveness", "low", 3.5, "Rational Engagement",
     "Low impulsiveness detected. Provide detailed comparisons, technical specs, and "
     "in-depth reviews. This customer makes deliberate, research-driven decisions."),
 
    ("social_media_dependency", "high", 6.5, "Social Commerce",
     "High social media dependency. Shift engagement to Instagram/Telegram. "
     "User-generated content (UGC) and peer reviews have outsized influence."),
 
    ("self_esteem", "low", 4.0, "Status & Identity",
     "Low self-esteem detected. Connect the brand to personal identity and values. "
     "VIP status tiers and exclusive member perks are particularly effective."),
 
    ("optimism_score", "low", 4.0, "Social Proof",
     "Low optimism. Lead with real customer success stories and tangible results "
     "rather than aspirational messaging."),
 
    ("life_satisfaction", "low", 4.0, "Experiential Products",
     "Low life satisfaction. Experiential offerings (travel, education, entertainment) "
     "outperform material products. Frame purchases as life improvements."),
 
    ("optimism_score", "high", 7.0, "Loyalty Programs",
     "High optimism. This customer is receptive to long-term loyalty programs, "
     "subscriptions, and tiered membership structures."),
]
 
SEGMENT_STRATEGIES: dict[str, str] = {
    "Champions":
        "Brand ambassador candidate. Activate referral program, early product access, "
        "and VIP events to maintain engagement.",
    "Loyal Customers":
        "High frequency, lower spend. Focus on upsell and cross-sell with product bundles "
        "and complementary item recommendations.",
    "Big Spenders":
        "High spend, low frequency. Increase purchase cadence with next-order discounts, "
        "smart reminders, and subscription options.",
    "Potential Loyalists":
        "Strong potential. Build habit with consistent weekly/monthly offers and "
        "gamification (points, tiers, milestones).",
    "Regular Customers":
        "Stable, mid-tier behavior. Personalize based on purchase history and introduce "
        "relevant educational content.",
    "At Risk / Low Value":
        "Low engagement. Deploy a single high-value win-back offer (large discount or "
        "free gift) to trigger re-engagement.",
}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINER
# ─────────────────────────────────────────────────────────────────────────────
 
class LoyaltyModelTrainer:
    """
    Trains two models on psychological features:
      1. Segment Classifier  — predicts Customer_Segment (6 classes)
      2. Spend Predictor     — predicts total_spent (continuous)
 
    Only PSYCH_FEATURES are used as model inputs so the app can run with
    psychological profile data alone (no purchase history needed).
    """
 
    def __init__(self, data: pd.DataFrame, model_dir: str = "models/"):
        self.console   = Console()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
 
        # Build RFM segments if not already present
        if "Customer_Segment" not in data.columns:
            self.console.print("⚙️  Customer_Segment not found — building RFM segments...")
            data = _build_rfm_segments(data)
            counts = data["Customer_Segment"].value_counts().to_dict()
            self.console.print(f"✓  Segments: {counts}")
 
        self.data = data.dropna(subset=PSYCH_FEATURES + ["Customer_Segment", "total_spent"]).copy()
 
        # Champions reference profile (used for gap analysis in the app)
        champions = self.data[self.data["Customer_Segment"] == "Champions"]
        self.champions_profile = (
            champions[PSYCH_FEATURES].mean().to_dict()
            if len(champions) > 0 else {f: 7.0 for f in PSYCH_FEATURES}
        )
 
        self.scaler     = StandardScaler()
        self.le         = LabelEncoder()
        # HistGradientBoosting: histogram-based implementation — 10-50x faster
        # than GradientBoosting on large datasets (>50k rows). Same accuracy.
        self.classifier = HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=50, random_state=42,
        )
        self.regressor  = HistGradientBoostingRegressor(
            max_iter=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=50, random_state=42,
        )
 
    def train(self, save_path: str | None = None) -> dict:
        X       = self.data[PSYCH_FEATURES]
        X_sc    = self.scaler.fit_transform(X)
        y_cls   = self.le.fit_transform(self.data["Customer_Segment"])
        y_reg   = self.data["total_spent"].values
 
        X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
            X_sc, y_cls, y_reg, test_size=0.2, random_state=42,
        )
 
        self.console.print(Panel(
            f"[bold {ACCENT}]Training Customer Loyalty Models[/bold {ACCENT}]",
            border_style=BORDER, padding=(0, 2),
        ))
 
        # ── Classifier ───────────────────────────────────────────────────────
        self.console.print(f"[{BORDER}]  Fitting segment classifier...[/]")
        self.classifier.fit(X_tr, yc_tr)
        cls_acc = self.classifier.score(X_te, yc_te)
        report  = classification_report(
            yc_te, self.classifier.predict(X_te),
            target_names=self.le.classes_, output_dict=True,
        )
        macro = report["macro avg"]
 
        # ── Regressor ────────────────────────────────────────────────────────
        self.console.print(f"[{BORDER}]  Fitting spend predictor...[/]")
        self.regressor.fit(X_tr, yr_tr)
        r2 = r2_score(yr_te, self.regressor.predict(X_te))
 
        # ── Per-segment accuracy table ────────────────────────────────────────
        report = classification_report(
            yc_te, self.classifier.predict(X_te),
            target_names=self.le.classes_, output_dict=True,
        )
        macro = report["macro avg"]
 
        perf = Table(
            title="Model Performance", box=ROUNDED,
            title_style=f"bold {ACCENT}", header_style=HEADER, border_style=BORDER,
        )
        perf.add_column("Model",  style="bold")
        perf.add_column("Metric", style=VALUE)
        perf.add_column("Score",  justify="right", style=ACCENT)
        perf.add_row("Segment Classifier", "Accuracy",        f"{cls_acc:.3f}")
        perf.add_row("",                   "F1 Macro",        f"{macro['f1-score']:.3f}")
        perf.add_row("",                   "Precision Macro", f"{macro['precision']:.3f}")
        perf.add_section()
        perf.add_row("Spend Predictor",    "R² Score",        f"{r2:.3f}")
        self.console.print(perf)
 
        # ── Save artifacts ────────────────────────────────────────────────────
        out = Path(save_path) if save_path else self.model_dir
        artifacts = {
            "classifier":        self.classifier,
            "regressor":         self.regressor,
            "scaler":            self.scaler,
            "label_encoder":     self.le,
            "champions_profile": self.champions_profile,
            "features":          PSYCH_FEATURES,
        }
        with open(out / "loyalty_model.pkl", "wb") as f:
            pickle.dump(artifacts, f)
 
        self.console.print(f"\n[{OK}]✓  Saved → {out}/loyalty_model.pkl[/]")
        return {"accuracy": cls_acc, "f1_macro": macro["f1-score"], "r2": r2}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE (used by app.py)
# ─────────────────────────────────────────────────────────────────────────────
 
class LoyaltyEngine:
    """
    Load trained artifacts and generate predictions + recommendations
    for a single customer profile.
    """
 
    def __init__(self, artifacts: dict):
        self.classifier       = artifacts["classifier"]
        self.regressor        = artifacts["regressor"]
        self.scaler           = artifacts["scaler"]
        self.le               = artifacts["label_encoder"]
        self.champions_profile= artifacts["champions_profile"]
        self.features         = artifacts["features"]
 
    @classmethod
    def load(cls, model_dir: str = "models/") -> "LoyaltyEngine":
        path = Path(model_dir) / "loyalty_model.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. "
                "Run LoyaltyModelTrainer(data).train() first."
            )
        with open(path, "rb") as f:
            return cls(pickle.load(f))
 
    def _recommendations(self, profile: dict[str, float], segment: str) -> list[str]:
        recs: list[str] = []
 
        # 1) Segment-level strategy (always shown)
        if segment in SEGMENT_STRATEGIES:
            recs.append(f"[Overall Strategy] {SEGMENT_STRATEGIES[segment]}")
 
        # 2) Rule-based feature triggers
        for feat, direction, threshold, tag, text in RULES:
            val = profile.get(feat, 5.0)
            hit = (direction == "high" and val > threshold) or \
                  (direction == "low"  and val < threshold)
            if hit:
                recs.append(f"[{tag}] {text}")
 
        # 3) Gap analysis — biggest distance from Champions profile
        pos_feats = ["self_esteem", "optimism_score", "life_satisfaction"]
        gaps      = {f: self.champions_profile.get(f, 5.0) - profile.get(f, 5.0)
                     for f in pos_feats}
        top_gap   = max(pos_feats, key=lambda f: gaps[f])
        if gaps[top_gap] > 1.5:
            recs.append(
                f"[Gap Analysis] '{top_gap.replace('_', ' ').title()}' is "
                f"{gaps[top_gap]:.1f} pts below the Champions average. "
                f"Messaging that compensates for this deficit will outperform generic campaigns."
            )
 
        return recs
 
    def predict(self, profile: dict[str, float]) -> dict:
        """
        Predict segment, spend, champion probability, and recommendations.
 
        Args:
            profile: dict of PSYCH_FEATURES → float (0–10).
                     Missing features default to 5.0.
 
        Returns:
            {segment, segment_probs, champion_prob, expected_spend, recommendations}
        """
        x       = np.array([profile.get(f, 5.0) for f in self.features]).reshape(1, -1)
        x_sc    = self.scaler.transform(x)
 
        # Segment
        seg_idx  = self.classifier.predict(x_sc)[0]
        segment  = self.le.inverse_transform([seg_idx])[0]
        probs    = self.classifier.predict_proba(x_sc)[0]
        seg_probs= {self.le.classes_[i]: round(float(p), 4) for i, p in enumerate(probs)}
        champ_p  = seg_probs.get("Champions", 0.0)
 
        # Spend
        exp_spend = max(0.0, float(self.regressor.predict(x_sc)[0]))
 
        return {
            "segment":         segment,
            "segment_probs":   seg_probs,
            "champion_prob":   champ_p,
            "expected_spend":  round(exp_spend, 2),
            "recommendations": self._recommendations(profile, segment),
        }
 