from chronos import BaseChronosPipeline, Chronos2Pipeline
import pandas as pd
import os
from dotenv import load_dotenv
import holidays

load_dotenv()

# Load the Chronos-2 pipeline
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu"
)

def regularize_hourly(g: pd.DataFrame) -> pd.DataFrame:
    sid = g[ID_COL].iloc[0] if ID_COL in g.columns else g.name
    g = g.sort_values(TS_COL)
    full_idx = pd.date_range(g[TS_COL].min(), g[TS_COL].max(), freq="h")
    g = g.set_index(TS_COL).reindex(full_idx)
    g.index.name = TS_COL
    g[ID_COL] = sid
    for col in TARGETS:
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce").ffill().bfill()
    return g.reset_index()

def add_holiday_flags(
    df: pd.DataFrame,
    ts_col: str = "ds",
    local_tz: str = "America/Montreal",
    observed: bool = True,
    include_names: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    if getattr(out[ts_col].dt, "tz", None) is not None:
        dates_for_calendar = out[ts_col].dt.tz_convert(local_tz).dt.date
    else:
        dates_for_calendar = out[ts_col].dt.date
    years_series = pd.Series(dates_for_calendar)
    years_series = years_series.dropna().map(lambda d: int(pd.Timestamp(d).year))
    if years_series.empty:
        raise ValueError("No valid datetimes found to extract holiday years.")
    years = list(range(int(years_series.min()), int(years_series.max()) + 1))
    qc_holidays = holidays.Canada(subdiv="QC", years=years, observed=observed)
    il_holidays = holidays.Israel(years=years, observed=observed)
    out["is_qc_holiday"] = [ ("yes" if d in qc_holidays else "no") if pd.notna(pd.Timestamp(d)) else "no" for d in dates_for_calendar ]
    out["is_jewish_holiday"] = [ ("yes" if d in il_holidays else "no") if pd.notna(pd.Timestamp(d)) else "no" for d in dates_for_calendar ]
    if include_names:
        out["qc_holiday_name"] = [ qc_holidays.get(d, "no") if pd.notna(pd.Timestamp(d)) else "no" for d in dates_for_calendar ]
        out["jewish_holiday_name"] = [ il_holidays.get(d, "no") if pd.notna(pd.Timestamp(d)) else "no" for d in dates_for_calendar ]
    return out

shift_types_dict = {
    'W1':'flow', 'X1':'pod', 'X3':'pod', 'X4':'vertical', 'X2':'vertical',
    'WOC1':'oncall', 'WOC2':'oncall', 'WOC3':'oncall', 'X5':'pod', 'W3':'overlap',
    'Y1':'pod', 'Y3':'pod', 'Y4':'vertical', 'Y2':'vertical', 'Y5':'pod',
    'Z1':'night', 'Z2':'night', 'D1':'pod', 'R1':'pod', 'P1':'vertical',
    'D2':'vertical', 'OC1':'oncall', 'OC2':'oncall', 'V1':'flow', 'A1':'pod',
    'G1':'vertical', 'E1':'pod', 'R2':'pod', 'A2':'pod', 'P2':'vertical',
    'E2':'vertical', 'N1':'night', 'N2':'night', 'L2':'overlap', 'L4':'overlap',
    'H1':'teaching', 'B1':'vertical', 'L1':'overlap', 'W5':'overlap', 'L6':'overlap', 'B2':'vertical'
}

# Load hourly data
df = pd.read_csv('https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&raw=1')
df.ds = pd.to_datetime(df.ds, errors="coerce")
df['id'] = 'jgh'

# Load shift data
all_shifts_df = pd.read_csv('https://www.dropbox.com/scl/fi/yeyr2a7pj6nry8i2q3m0c/all_shifts.csv?rlkey=q1su2h8fqxfnlu7t1l2qe1w0q&raw=1')
all_shifts_df['shift_start'] = pd.to_datetime(all_shifts_df['shift_start']).dt.round('h')
all_shifts_df['shift_end'] = pd.to_datetime(all_shifts_df['shift_end']).dt.round('h')
all_shifts_df['shift_type'] = all_shifts_df['shift_short_name'].map(shift_types_dict)

expanded_rows = []
for _, row in all_shifts_df.iterrows():
    hours = pd.date_range(row['shift_start'], row['shift_end'], freq='h', inclusive='left')
    for h in hours:
        expanded_rows.append({'ds': h, 'user': row['first_name']+row['last_name'], 'shift_type': row['shift_type'], 'shift_short_name': row['shift_short_name']})

expanded_df = pd.DataFrame(expanded_rows)
hourly_shifts_by_user_df = expanded_df.pivot_table(index='ds', columns='user', values='shift_type', aggfunc='first').fillna('NotWorking')

ID_COL = "id"
TS_COL = "ds"
TARGETS = ["oncall_busy"]

# Load On-Call Busy Labels
oncall_labels = pd.read_csv('../hourly_oncall_used_for_busy_since_2022.csv')
oncall_labels['ds'] = pd.to_datetime(oncall_labels['ds'])
oncall_labels = oncall_labels.rename(columns={'oncall-used-for-busy': 'oncall_busy'})

# Merge on-call labels into main df
df = df.merge(oncall_labels, on='ds', how='left').fillna({'oncall_busy': 0})
df['oncall_busy'] = df['oncall_busy'].astype(float)

df = df.copy()
df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")
df = df.dropna(subset=[TS_COL])
df[TS_COL] = df[TS_COL].dt.floor("h")
df = df.sort_values([ID_COL, TS_COL]).drop_duplicates([ID_COL, TS_COL], keep="last")

gb = df.groupby(ID_COL, group_keys=False)
try:
    df = gb.apply(regularize_hourly, include_groups=False)
except TypeError:
    df = gb.apply(regularize_hourly)

# All variables setup
df_with_staffing = df.merge(hourly_shifts_by_user_df, on='ds')
weather_df = pd.read_csv('https://www.dropbox.com/scl/fi/gmhwwld9z9yychg4r0yuk/weather.csv?rlkey=66c78m90aviamr0x0uu72pfr8&raw=1')
weather_df.ds = pd.to_datetime(weather_df.ds, errors="coerce")

all_variable_df = add_holiday_flags(df_with_staffing, ts_col='ds', include_names=True).merge(weather_df, on='ds')

# Future DF Preparation
future_df_staffing = hourly_shifts_by_user_df.reset_index()[hourly_shifts_by_user_df.reset_index()['ds'] > df['ds'].max()].head(24)
future_df_staffing['id'] = 'jgh'
future_weather_df = weather_df[weather_df.ds > df.ds.max()].head(24)
future_weather_df['id'] = 'jgh'

# Merge future features
future_df_base = future_df_staffing.merge(future_weather_df, on=['ds', 'id'])
future_df_base = add_holiday_flags(future_df_base, ts_col='ds', include_names=True)

# Ensure common columns match for the pipeline
common_columns = [col for col in future_df_base.columns if col in all_variable_df.columns]
future_df_base = future_df_base[common_columns]

# Predict On-Call Need
print('Predicting On-Call Need for the next 24 hours...')
forecast = pipeline.predict_df(
    all_variable_df,
    prediction_length=24,
    future_df=future_df_base,
    id_column=ID_COL,
    timestamp_column=TS_COL,
    target=TARGETS,
    quantile_levels=[0.5]
)

# Process and save
res = forecast[['ds', 'target_name', 'predictions']].rename(columns={'predictions': 'predicted_oncall_prob'})
res.to_csv('oncall_need_forecast.csv', index=False)
print('Saved oncall_need_forecast.csv')
