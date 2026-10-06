from staffing_features import build_legacy_staffing_identity
from chronos import BaseChronosPipeline, Chronos2Pipeline
import pandas as pd
import os
from dotenv import load_dotenv
import holidays
from oncall_labels import merge_activation_labels

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
            g[col] = pd.to_numeric(g[col], errors="coerce")
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

# Load hourly data
df = pd.read_csv('https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&raw=1')
df.ds = pd.to_datetime(df.ds, errors="coerce")
df['id'] = 'jgh'

# Load shift data and use the same effective-dated roles as the current models.
all_shifts_df = pd.read_csv('https://www.dropbox.com/scl/fi/yeyr2a7pj6nry8i2q3m0c/all_shifts.csv?rlkey=q1su2h8fqxfnlu7t1l2qe1w0q&raw=1')
hourly_shifts_by_user_df = build_legacy_staffing_identity(all_shifts_df)

ID_COL = "id"
TS_COL = "ds"
TARGETS = ["oncall_busy"]

# Load On-Call Busy Labels
oncall_labels = pd.read_csv('../hourly_oncall_used_for_busy_since_2022.csv')
oncall_labels['ds'] = pd.to_datetime(oncall_labels['ds'])
# Merge on-call labels into main df
df = merge_activation_labels(df, oncall_labels).rename(columns={'oncall_active': 'oncall_busy'})
df['oncall_busy'] = df['oncall_busy'].astype(float)
if df['oncall_busy'].isna().any():
    raise ValueError('Legacy on-call forecast requires complete observed activation labels; missing is unknown.')

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

for column in hourly_shifts_by_user_df.columns.intersection(future_df_base.columns):
    seen = set(all_variable_df[column].dropna().astype(str))
    future_df_base.loc[~future_df_base[column].astype(str).isin(seen), column] = 'NotWorking'

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
