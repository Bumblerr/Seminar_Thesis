import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class FeatureEngineer:

    def __init__(self):
        pass
        
    def create_features(self, df):
        
        # Gap Ahead = Differenz der LapStartTime zum Vordermann (Position-1)
        df = df.sort_values(["GP", "Year", "LapNumber", "Position"]).copy()
        df["gapAheadSeconds"] = df.groupby(["GP", "Year", "LapNumber"])["LapStartTime"] \
                                .diff() \
                                .dt.total_seconds() \
                                .fillna(0)
        
        
        # Sector Deltas
        df["s1_delta"] = (
            df.sort_values(["GP", "Year", "Driver", "LapNumber"])
                .groupby(["GP", "Year", "Driver"])["Sector1Time"]
                .diff()
                .dt.total_seconds()
                .fillna(0)
        )

        df["s2_delta"] = (
            df.sort_values(["GP", "Year", "Driver", "LapNumber"])
                .groupby(["GP", "Year", "Driver"])["Sector2Time"]
                .diff()
                .dt.total_seconds()
                .fillna(0)
        )

        df["s3_delta"] = (
            df.sort_values(["GP", "Year", "Driver", "LapNumber"])
                .groupby(["GP", "Year", "Driver"])["Sector3Time"]
                .diff()
                .dt.total_seconds()
                .fillna(0)
        )
        
        # Lap Pace Delta
        df["lap_pace_delta"] = (
            df.sort_values(["GP", "Year", "Driver", "LapNumber"])
                .groupby(["GP", "Year", "Driver"])["LapTime"]
                .diff()
                .dt.total_seconds()
                .fillna(0)
        )
        
        
        # # Compounds
        # compund_mapping = {
        #     'SOFT': 0,
        #     'MEDIUM': 1,
        #     'HARD': 2,
        #     'INTERMEDIATE': 3,
        #     'WET': 4
        # }
        # df['compoundType'] = df['Compound'].map(compund_mapping)
        
        # # Track Status
        # df['TrackStatus'] = df['TrackStatus'].to_numpy().astype(int)
        
        df = pd.get_dummies(
            df,
            columns=["Compound", "TrackStatus"],
            prefix=["compound", "status"],
            drop_first=False  # wichtig: für XGBoost und LR alle Spalten behalten
        )
        
        
        # Rainfall
        df['is_raining'] = df['Rainfall'].apply(lambda x: 1 if x > 0 else 0)
        
        # Position Change
        df["position_change"] = (
            df.sort_values(["GP", "Year", "Driver", "LapNumber"])
                .groupby(["GP", "Year", "Driver"])["Position"]
                .diff()
                .fillna(0)
        )
        
        # Pit This Lap
        df['pit_this_lap'] = df['PitInTime'].notna().astype(int)
        
        #Pit in next lap
        df['pit_in_next_lap'] = (
            df.groupby(["GP", "Year", "Driver"])["pit_this_lap"]
                .transform(lambda x: (x.shift(-1).fillna(0)))
                .clip(0, 1)
                .astype(int)
        )
        
        # Pit in next three laps
        df['pit_in_next_three_laps'] = (
            df.groupby(["GP", "Year", "Driver"])["pit_this_lap"]
                .transform(lambda x:
                            (x.shift(-1).fillna(0) +
                            x.shift(-2).fillna(0) +
                            x.shift(-3).fillna(0)))
                .clip(0, 1)
                .astype(int)
        )
        
        # Pit in next five laps
        df['pit_in_next_five_laps'] = (
            df.groupby(["GP", "Year", "Driver"])["pit_this_lap"]
                .transform(lambda x:
                            (x.shift(-1).fillna(0) +
                            x.shift(-2).fillna(0) +
                            x.shift(-3).fillna(0) + 
                            x.shift(-4).fillna(0) +
                            x.shift(-5).fillna(0)))
                .clip(0, 1)
                .astype(int)
        )
        return df
    
    def choose_features(self, df):
        selected_features = [
            "gapAheadSeconds",
            "s1_delta",
            "s2_delta",
            "s3_delta",
            "lap_pace_delta",
            "is_raining",
            "position_change"
        ]
        # + alle One-Hot-Spalten
        selected_features += [col for col in df.columns if col.startswith("compound_")]
        selected_features += [col for col in df.columns if col.startswith("status_")]
        
        return selected_features
    
    def split_features(self, df, features):
        X = df[features]
        y = df["pit_in_next_lap"]
        
        return X, y