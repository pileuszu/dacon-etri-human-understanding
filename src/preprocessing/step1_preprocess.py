import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
from scipy.stats import entropy
import warnings

warnings.filterwarnings('ignore')

class MultiModalPreprocessor:
    def __init__(self, data_dir='data/ch2025_data_items', metrics_path='data/ch2026_metrics_train.csv'):
        self.data_dir = data_dir
        self.metrics_path = metrics_path
        self.subject_date_key = ['subject_id', 'date']
        
    def load_metrics(self):
        print("Loading metrics...")
        df = pd.read_csv(self.metrics_path)
        df['date'] = pd.to_datetime(df['lifelog_date']).dt.date
        return df

    def safe_read_parquet(self, filename):
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: {filename} not found.")
            return None
        df = pd.read_parquet(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
        return df

    def aggregate_ac_status(self):
        print("Processing AC Status...")
        df = self.safe_read_parquet('ch2025_mACStatus.parquet')
        if df is None: return None
        
        # Missingness indicator as suggested in RESEARCH_METHODOLOGY.md
        agg = df.groupby(self.subject_date_key).agg(
            charging_ratio=('m_charging', 'mean'),
            charging_count=('m_charging', 'sum'),
            charging_events=('m_charging', lambda x: (x.diff() == 1).sum())
        ).reset_index()
        return agg

    def aggregate_activity(self):
        print("Processing Activity...")
        df = self.safe_read_parquet('ch2025_mActivity.parquet')
        if df is None: return None
        
        # Activity distribution and entropy
        def get_activity_stats(x):
            counts = x.value_counts(normalize=True)
            ent = entropy(counts)
            most_freq = counts.idxmax() if not counts.empty else -1
            return pd.Series([ent, most_freq], index=['activity_entropy', 'most_freq_activity'])

        agg_stats = df.groupby(self.subject_date_key)['m_activity'].apply(get_activity_stats).unstack().reset_index()
        
        # Duration of each activity (simplified as count ratio)
        activity_types = [0, 1, 2, 3, 4, 5, 7, 8]
        for act in activity_types:
            df[f'act_{act}'] = (df['m_activity'] == act).astype(int)
        
        agg_dur = df.groupby(self.subject_date_key)[[f'act_{act}' for act in activity_types]].mean().reset_index()
        return pd.merge(agg_stats, agg_dur, on=self.subject_date_key)

    def aggregate_gps(self):
        print("Processing GPS...")
        df = self.safe_read_parquet('ch2025_mGps.parquet')
        if df is None: return None
        
        # Extract speed from list if it's nested, or use the column
        # Based on inspection, columns are ['subject_id', 'timestamp', 'm_gps']
        # m_gps looks like List[Coord, Speed] or similar. Let's assume it's pre-processed or we need to extract.
        # If it's a list of dicts/lists, we extract 'speed'.
        
        def extract_gps_features(x):
            speeds = []
            for item_list in x:
                if isinstance(item_list, (list, np.ndarray)):
                    for item in item_list:
                        if isinstance(item, dict) and 'speed' in item:
                            speeds.append(item['speed'])
                        elif isinstance(item, (int, float)):
                            speeds.append(item)
                elif isinstance(item_list, (int, float)):
                    speeds.append(item_list)
            
            if not speeds: return pd.Series([0, 0, 0], index=['gps_speed_mean', 'gps_speed_max', 'gps_count'])
            
            speeds = np.array([float(s) for s in speeds if s is not None])
            if len(speeds) == 0: return pd.Series([0, 0, 0], index=['gps_speed_mean', 'gps_speed_max', 'gps_count'])

            # Unit Normalization: km/h to m/s if > 100
            speeds = np.where(speeds > 100, speeds / 3.6, speeds)
            
            return pd.Series([np.mean(speeds), np.max(speeds), len(speeds)], 
                             index=['gps_speed_mean', 'gps_speed_max', 'gps_count'])

        agg = df.groupby(self.subject_date_key)['m_gps'].apply(extract_gps_features).unstack().reset_index()
        return agg

    def aggregate_heart_rate(self):
        print("Processing Heart Rate...")
        df = self.safe_read_parquet('ch2025_wHr.parquet')
        if df is None: return None
        
        def process_hr_list(x):
            all_hr = []
            for hr_list in x:
                if isinstance(hr_list, (list, np.ndarray)):
                    all_hr.extend(hr_list)
                else:
                    all_hr.append(hr_list)
            if not all_hr: return pd.Series([0, 0, 0], index=['hr_mean', 'hr_std', 'hr_count'])
            return pd.Series([np.mean(all_hr), np.std(all_hr), len(all_hr)], 
                             index=['hr_mean', 'hr_std', 'hr_count'])

        agg = df.groupby(self.subject_date_key)['heart_rate'].apply(process_hr_list).unstack().reset_index()
        return agg

    def aggregate_light(self, filename, prefix):
        print(f"Processing Light ({prefix})...")
        df = self.safe_read_parquet(filename)
        if df is None: return None
        
        col = 'm_light' if 'm_light' in df.columns else 'w_light'
        
        # Circadian Rhythm-based Weighting: weight evening light more for sleep quality impact
        df['weight'] = df['hour'].apply(lambda h: 2.0 if 20 <= h or h <= 4 else 1.0)
        df['weighted_light'] = df[col] * df['weight']
        
        agg = df.groupby(self.subject_date_key).agg(
            light_mean=(col, 'mean'),
            light_max=(col, 'max'),
            light_weighted_mean=('weighted_light', 'mean')
        ).reset_index()
        agg.columns = [c if c in self.subject_date_key else f'{prefix}_{c}' for c in agg.columns]
        return agg

    def aggregate_pedo(self):
        print("Processing Pedo...")
        df = self.safe_read_parquet('ch2025_wPedo.parquet')
        if df is None: return None
        
        agg = df.groupby(self.subject_date_key).agg(
            total_steps=('step', 'sum'),
            avg_step_freq=('step_frequency', 'mean'),
            total_distance=('distance', 'sum'),
            avg_speed=('speed', 'mean'),
            total_calories=('burned_calories', 'sum')
        ).reset_index()
        return agg

    def run(self):
        metrics = self.load_metrics()
        
        # Aggregators
        aggs = [
            self.aggregate_ac_status(),
            self.aggregate_activity(),
            self.aggregate_gps(),
            self.aggregate_heart_rate(),
            self.aggregate_light('ch2025_mLight.parquet', 'phone'),
            self.aggregate_light('ch2025_wLight.parquet', 'watch'),
            self.aggregate_pedo()
        ]
        
        final_df = metrics
        for agg in aggs:
            if agg is not None:
                final_df = pd.merge(final_df, agg, on=self.subject_date_key, how='left')
        
        # Null-mask Embedding: Create binary indicators for missing sensor data
        for agg, name in zip(aggs, ['ac', 'act', 'gps', 'hr', 'm_light', 'w_light', 'pedo']):
            if agg is not None:
                final_df[f'is_missing_{name}'] = final_df[f'{agg.columns[-1]}'].isna().astype(int)
        
        # Fill remaining NaNs with 0 (or appropriate value)
        final_df = final_df.fillna(0)
        
        # Save
        output_path = 'data/processed/step1_processed_train.csv'
        final_df.to_csv(output_path, index=False)
        print(f"Preprocessing complete. Saved to {output_path}")
        return final_df

if __name__ == "__main__":
    preprocessor = MultiModalPreprocessor()
    preprocessor.run()
