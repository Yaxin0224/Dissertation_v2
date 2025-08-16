#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inner London LSOA GVI Prediction Model (English outputs)

- Fills missing/insufficient LSOA GVI using a RandomForest model trained on completed rows.
- All console messages, report contents, and chart text are in English.
- More robust to missing columns, datatypes, non-GUI environments, and path issues.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Use a non-interactive backend so saving figures works on any machine
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # Optional
except Exception:
    sns = None

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -----------------------------
# Config (BASE_DIR is overridable via CLI arg)
# -----------------------------
if len(sys.argv) > 1:
    BASE_DIR = Path(sys.argv[1]).expanduser().resolve()
else:
    BASE_DIR = Path(
        r"C:\Users\z1782\OneDrive - University College London\Attachments\004\methodology\Dissertation_v2"
    )

OUTPUT_DIR = BASE_DIR / "output" / "full_london_gvi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = OUTPUT_DIR / "inner_london_all_lsoas_status.csv"
OUTPUT_FILE = OUTPUT_DIR / "inner_london_all_lsoas_complete.csv"
REPORT_FILE = OUTPUT_DIR / "prediction_report.txt"
PLOT_FILE = OUTPUT_DIR / "gvi_prediction_analysis.png"

# Fonts: prefer SimHei for Chinese compatibility, fallback to DejaVu Sans (safe on most systems)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class GVIPredictionModel:
    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.model: RandomForestRegressor | None = None
        self.feature_cols: list[str] | None = None
        self.le_borough = LabelEncoder()

    # ---------- Data ----------
    def load_data(self):
        print("=" * 70)
        print("Inner London LSOA GVI Prediction Model")
        print("=" * 70)

        print("\n1) Loading data...")
        if not INPUT_FILE.exists():
            raise FileNotFoundError(
                f"Input file not found: {INPUT_FILE}\n"
                f"- Confirm the path exists, OR pass your base dir as an argument:\n"
                f'  python gvi_prediction_model.py "D:\\MyProject\\Dissertation_v2"'
            )

        self.df = pd.read_csv(INPUT_FILE, encoding_errors="ignore")
        self.df.columns = [c.strip() for c in self.df.columns]

        required = [
            "lsoa_code",
            "lsoa_name",
            "borough",
            "imd_score",
            "imd_quintile",
            "n_images",
            "status",
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(missing)
                + "\nColumn purposes:\n"
                "- lsoa_code/lsoa_name: unique IDs\n"
                "- borough: borough name\n"
                "- imd_score/imd_quintile: deprivation indicators\n"
                "- n_images: number of available images\n"
                "- status: completed/insufficient/no_data"
            )

        # Create optional columns if absent
        for col in [
            "mean_gvi",
            "median_gvi",
            "std_gvi",
            "min_gvi",
            "max_gvi",
            "q25_gvi",
            "q75_gvi",
        ]:
            if col not in self.df.columns:
                self.df[col] = np.nan

        # Type corrections
        self.df["imd_score"] = pd.to_numeric(self.df["imd_score"], errors="coerce")
        self.df["n_images"] = (
            pd.to_numeric(self.df["n_images"], errors="coerce").fillna(0).clip(lower=0)
        )
        self.df["borough"] = self.df["borough"].astype(str).fillna("Unknown").str.strip()
        self.df["status"] = self.df["status"].astype(str).str.strip().str.lower()

        # Summary
        print(f"   Total LSOAs: {len(self.df)}")
        print(f"   Completed: {len(self.df[self.df['status'] == 'completed'])}")
        print(f"   Insufficient: {len(self.df[self.df['status'] == 'insufficient'])}")
        print(f"   No data: {len(self.df[self.df['status'] == 'no_data'])}")

        return self.df

    # ---------- Features ----------
    def feature_engineering(self):
        print("\n2) Feature engineering...")

        # Borough encoding
        self.df["borough_encoded"] = self.le_borough.fit_transform(self.df["borough"])

        completed = self.df[self.df["status"] == "completed"].copy()
        if completed.empty:
            raise ValueError(
                "No 'completed' rows available to train the model. Please check the dataset."
            )

        # Borough mean GVI (from completed only)
        borough_mean = completed.groupby("borough")["mean_gvi"].mean()
        self.df["mean_gvi_borough"] = self.df["borough"].map(borough_mean)

        # imd_quintile encoding: map first, fallback to imd_score quantiles
        quintile_map = {"Q1_Least": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Q5_Most": 5}
        q_raw = self.df["imd_quintile"].astype(str).str.strip()
        q_code = q_raw.map(quintile_map)

        if q_code.isna().any():
            try:
                bins = np.nanpercentile(self.df["imd_score"], [0, 20, 40, 60, 80, 100])
                self.df["imd_quintile_encoded"] = (
                    np.digitize(self.df["imd_score"], bins[1:-1], right=True) + 1
                )
            except Exception:
                self.df["imd_quintile_encoded"] = 3  # fallback to median bucket
        else:
            self.df["imd_quintile_encoded"] = q_code

        # imd_quintile mean GVI (from completed only)
        quintile_mean = completed.groupby("imd_quintile")["mean_gvi"].mean()
        self.df["mean_gvi_quintile"] = self.df["imd_quintile"].map(quintile_mean)

        # Spatial feature (simple: within-borough nearest 5 by IMD score)
        self.df = self._add_spatial_features()

        # Interaction feature
        self.df["imd_borough_interaction"] = (
            self.df["imd_score"] * self.df["borough_encoded"]
        )

        # Image availability (normalized)
        max_img = max(100.0, float(self.df["n_images"].max() or 0))
        self.df["image_availability_score"] = (self.df["n_images"] / max_img).fillna(0)

        print("   Created 7 feature variables.")
        self.feature_cols = [
            "imd_score",
            "borough_encoded",
            "mean_gvi_borough",
            "mean_gvi_quintile",
            "imd_quintile_encoded",
            "spatial_lag_gvi",
            "imd_borough_interaction",
            "image_availability_score",
        ]
        return self.df

    def _add_spatial_features(self):
        spatial_lag = []
        global_mean = self.df[self.df["status"] == "completed"]["mean_gvi"].mean()

        for _, row in self.df.iterrows():
            same_borough = self.df[
                (self.df["borough"] == row["borough"])
                & (self.df["status"] == "completed")
                & (self.df["lsoa_code"] != row["lsoa_code"])
            ].copy()

            if len(same_borough) > 0:
                same_borough["imd_diff"] = (same_borough["imd_score"] - row["imd_score"]).abs()
                nearest = same_borough.nsmallest(5, "imd_diff")
                spatial_lag.append(
                    nearest["mean_gvi"].mean() if len(nearest) > 0 else global_mean
                )
            else:
                spatial_lag.append(global_mean)

        self.df["spatial_lag_gvi"] = spatial_lag
        return self.df

    # ---------- Train ----------
    def train_model(self):
        print("\n3) Training RandomForest model...")

        train_data = self.df[self.df["status"] == "completed"].copy()

        # Fill missing numerics with medians
        for col in self.feature_cols:
            if train_data[col].isna().any():
                train_data[col] = train_data[col].fillna(train_data[col].median())

        X_train = train_data[self.feature_cols]
        y_train = pd.to_numeric(train_data["mean_gvi"], errors="coerce")
        keep = y_train.notna()
        X_train, y_train = X_train[keep], y_train[keep]

        print(f"   Training samples: {len(X_train)}")

        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

        if len(X_train) >= 10:
            print("\n   5-fold cross validation...")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=kf, scoring="r2")
            print(f"   CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
            print(f"   Fold scores: {cv_scores}")
        else:
            print("   Not enough samples for CV; skipping.")

        self.model.fit(X_train, y_train)

        fi = pd.Series(self.model.feature_importances_, index=self.feature_cols).sort_values(
            ascending=False
        )
        print("\n   Feature importance:")
        for f, v in fi.items():
            print(f"     {f}: {v:.3f}")

        return self.model

    # ---------- Predict ----------
    def predict_missing(self):
        print("\n4) Predicting missing/insufficient LSOAs...")
        missing_mask = self.df["status"].isin(["insufficient", "no_data"])
        missing_data = self.df[missing_mask].copy()
        print(f"   To predict: {len(missing_data)} LSOAs")

        if len(missing_data) == 0:
            print("   Nothing to predict.")
            return

        # Fill using medians from completed rows
        train_medians = (
            self.df[self.df["status"] == "completed"][self.feature_cols]
            .median(numeric_only=True)
        )
        for col in self.feature_cols:
            missing_data[col] = missing_data[col].fillna(train_medians.get(col, 0))

        X_missing = missing_data[self.feature_cols]
        predictions = self.model.predict(X_missing)

        # Write back
        self.df.loc[missing_mask, "mean_gvi"] = predictions
        self.df.loc[missing_mask, "predicted"] = True

        # Estimate other distributional stats from mean
        for idx in self.df.index[missing_mask]:
            mean_val = float(self.df.at[idx, "mean_gvi"])
            self.df.at[idx, "median_gvi"] = mean_val
            self.df.at[idx, "std_gvi"] = 8.0
            self.df.at[idx, "min_gvi"] = max(0.0, mean_val - 15.0)
            self.df.at[idx, "max_gvi"] = min(100.0, mean_val + 15.0)
            self.df.at[idx, "q25_gvi"] = max(0.0, mean_val - 5.0)
            self.df.at[idx, "q75_gvi"] = min(100.0, mean_val + 5.0)

        print("\n   Prediction summary:")
        print(f"   - Mean predicted GVI: {np.mean(predictions):.2f}%")
        print(f"   - Range: {np.min(predictions):.2f}% - {np.max(predictions):.2f}%")

        for status in ["insufficient", "no_data"]:
            sdat = self.df[self.df["status"] == status]
            if len(sdat) > 0:
                print(
                    f"   - {status}: {len(sdat)} rows, mean GVI={sdat['mean_gvi'].mean():.2f}%"
                )

    # ---------- Validate ----------
    def validate_predictions(self):
        print("\n5) Validating (leave-one-out style on a sample)...")
        complete = self.df[self.df["status"] == "completed"]
        n = min(50, len(complete))
        if n < 5:
            print("   Too few completed rows (<5); skipping validation.")
            return

        sample = complete.sample(n=n, random_state=42)
        preds, acts = [], []

        for idx, row in sample.iterrows():
            temp_train = self.df[
                (self.df["status"] == "completed") & (self.df.index != idx)
            ]
            X_temp = temp_train[self.feature_cols]
            y_temp = pd.to_numeric(temp_train["mean_gvi"], errors="coerce")
            keep = y_temp.notna()
            X_temp, y_temp = X_temp[keep], y_temp[keep]

            temp_model = RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1
            )
            temp_model.fit(X_temp, y_temp)

            X_test = row[self.feature_cols].values.reshape(1, -1)
            preds.append(temp_model.predict(X_test)[0])
            acts.append(row["mean_gvi"])

        r2 = r2_score(acts, preds)
        mae = mean_absolute_error(acts, preds)
        rmse = np.sqrt(mean_squared_error(acts, preds))

        print(f"   Validation ({n} samples):")
        print(f"   - R²: {r2:.3f}")
        print(f"   - MAE: {mae:.2f}%")
        print(f"   - RMSE: {rmse:.2f}%")

    # ---------- Save ----------
    def save_results(self):
        print("\n6) Saving results...")

        self.df["data_source"] = np.where(
            self.df["status"].isin(["insufficient", "no_data"]), "predicted", "measured"
        )

        output_cols = [
            "lsoa_code",
            "lsoa_name",
            "borough",
            "imd_score",
            "imd_quintile",
            "n_images",
            "mean_gvi",
            "median_gvi",
            "std_gvi",
            "min_gvi",
            "max_gvi",
            "q25_gvi",
            "q75_gvi",
            "status",
            "data_source",
        ]
        output_cols = [c for c in output_cols if c in self.df.columns]

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.df[output_cols].to_csv(OUTPUT_FILE, index=False)
        print(f"   Complete dataset saved to: {OUTPUT_FILE}")

        self._generate_report()

    def _generate_report(self):
        report = []
        report.append("=" * 70)
        report.append("Inner London LSOA GVI Prediction Report")
        report.append("=" * 70)

        # Overall stats
        report.append("\n[Overall]")
        n_total = len(self.df)
        n_meas = (self.df["data_source"] == "measured").sum()
        n_pred = (self.df["data_source"] == "predicted").sum()
        report.append(f"Total LSOAs: {n_total}")
        report.append(f"Measured: {n_meas} ({n_meas / n_total * 100:.1f}%)")
        report.append(f"Predicted: {n_pred} ({n_pred / n_total * 100:.1f}%)")

        # GVI stats
        report.append("\n[GVI Stats]")
        report.append(
            f"Overall mean GVI: {self.df['mean_gvi'].mean():.2f}% ± {self.df['mean_gvi'].std():.2f}%"
        )
        report.append(
            f"Measured mean GVI: {self.df[self.df['data_source']=='measured']['mean_gvi'].mean():.2f}%"
        )
        report.append(
            f"Predicted mean GVI: {self.df[self.df['data_source']=='predicted']['mean_gvi'].mean():.2f}%"
        )

        # By borough
        report.append("\n[By Borough]")
        borough_stats = (
            self.df.groupby("borough")
            .agg(
                mean_gvi=("mean_gvi", "mean"),
                total_lsoas=("lsoa_code", "count"),
                predicted=("data_source", lambda x: (x == "predicted").sum()),
            )
            .round(2)
        )
        borough_stats = borough_stats.rename(
            columns={"mean_gvi": "Mean GVI", "total_lsoas": "LSOAs", "predicted": "Predicted"}
        )
        report.append(borough_stats.to_string())

        # By IMD quintile
        report.append("\n[By IMD Quintile]")
        quintile_stats = (
            self.df.groupby("imd_quintile")
            .agg(
                mean_gvi=("mean_gvi", "mean"),
                total_lsoas=("lsoa_code", "count"),
                predicted=("data_source", lambda x: (x == "predicted").sum()),
            )
            .round(2)
        )
        quintile_stats = quintile_stats.rename(
            columns={"mean_gvi": "Mean GVI", "total_lsoas": "LSOAs", "predicted": "Predicted"}
        )
        report.append(quintile_stats.to_string())

        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(f"   Detailed report saved to: {REPORT_FILE}")

    # ---------- Plots ----------
    def create_visualizations(self):
        print("\n7) Creating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Distribution: measured vs predicted
        ax1 = axes[0, 0]
        measured = self.df[self.df["data_source"] == "measured"]["mean_gvi"].dropna()
        predicted = self.df[self.df["data_source"] == "predicted"]["mean_gvi"].dropna()
        ax1.hist([measured, predicted], bins=30, label=["Measured", "Predicted"], alpha=0.7)
        ax1.set_xlabel("GVI (%)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("GVI Distribution: Measured vs Predicted")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Average GVI by borough
        ax2 = axes[0, 1]
        borough_gvi = self.df.groupby("borough")["mean_gvi"].mean().sort_values()
        borough_gvi.plot(kind="barh", ax=ax2, alpha=0.8)
        ax2.set_xlabel("Mean GVI (%)")
        ax2.set_title("Average GVI by Borough")
        ax2.grid(True, alpha=0.3)

        # 3. Scatter: GVI vs IMD score
        ax3 = axes[1, 0]
        measured_df = self.df[self.df["data_source"] == "measured"]
        predicted_df = self.df[self.df["data_source"] == "predicted"]
        ax3.scatter(
            measured_df["imd_score"],
            measured_df["mean_gvi"],
            alpha=0.5,
            s=20,
            label="Measured",
        )
        ax3.scatter(
            predicted_df["imd_score"],
            predicted_df["mean_gvi"],
            alpha=0.7,
            s=30,
            label="Predicted",
            marker="^",
        )
        ax3.set_xlabel("IMD Score")
        ax3.set_ylabel("GVI (%)")
        ax3.set_title("GVI vs IMD Score")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Data completeness by borough
        ax4 = axes[1, 1]
        completeness = (
            self.df.groupby("borough")["data_source"]
            .apply(lambda x: (x == "measured").mean() * 100)
            .sort_values()
        )
        completeness.plot(kind="barh", ax=ax4, alpha=0.8)
        ax4.set_xlabel("Data Completeness (%)")
        ax4.set_title("Percentage of Measured GVI by Borough")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=300, bbox_inches="tight")
        print(f"   Figure saved to: {PLOT_FILE}")

    # ---------- Pipeline ----------
    def run(self):
        try:
            self.load_data()
            self.feature_engineering()
            self.train_model()
            self.predict_missing()
            self.validate_predictions()
            self.save_results()
            self.create_visualizations()

            print("\n" + "=" * 70)
            print("✅ Prediction complete!")
            print("=" * 70)

            df = self.df
            print("\nFinal dataset:")
            print(f"  - Total LSOAs: {len(df)}")
            print(f"  - All have GVI values: {df['mean_gvi'].notna().all()}")
            print(f"  - Overall mean GVI: {df['mean_gvi'].mean():.2f}%")
            print(
                f"  - Range: {df['mean_gvi'].min():.2f}% - {df['mean_gvi'].max():.2f}%"
            )

            print("\nOutputs:")
            print(f"  - Complete data: {OUTPUT_FILE.name}")
            print(f"  - Report: {REPORT_FILE.name}")
            print(f"  - Visualization: {PLOT_FILE.name}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback

            traceback.print_exc()


def main():
    predictor = GVIPredictionModel()
    predictor.run()


if __name__ == "__main__":
    main()
