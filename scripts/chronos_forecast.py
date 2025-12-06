from chronos import BaseChronosPipeline, Chronos2Pipeline
import pandas as pd
import os
from dotenv import load_dotenv
import requests
from utils import upload
import dropbox
from pandas.tseries.frequencies import to_offset
import holidays

load_dotenv()

# Load the Chronos-2 pipeline
# GPU recommended for faster inference, but CPU is also supported
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    # device_map="cuda"
    device_map="cpu"
)

def regularize_hourly(g: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex each group's timestamps to strict hourly and fill gaps.
    Works whether the grouping column is present or omitted (include_groups=False).
    """
    # The group key (id) is available as g.name; if ID_COL exists, prefer it.
    sid = g[ID_COL].iloc[0] if ID_COL in g.columns else g.name

    g = g.sort_values(TS_COL)
    full_idx = pd.date_range(g[TS_COL].min(), g[TS_COL].max(), freq="h")
    g = g.set_index(TS_COL).reindex(full_idx)
    g.index.name = TS_COL

    # restore id (constant for the whole group)
    g[ID_COL] = sid

    # numeric + fill for targets
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
    """
    Adds boolean columns:
      • is_qc_holiday       — Québec public holiday (CA-QC)
      • is_jewish_holiday   — Israeli public/Jewish holiday (IL)
    Optionally adds:
      • qc_holiday_name
      • jewish_holiday_name

    Notes:
      • Holiday checks are date-based (00:00–24:00 local calendar date),
        not sundown-to-sundown observance.
      • NaT timestamps are ignored gracefully.
    """
    out = df.copy()

    # 1) Parse to datetime
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")

    # 2) Get the calendar DATE to use for holiday lookup
    #    - If tz-aware: convert to Montreal then take .date
    #    - If naive: assume values already represent local Montreal wall-clock; just take .date
    if getattr(out[ts_col].dt, "tz", None) is not None:
        dates_for_calendar = out[ts_col].dt.tz_convert(local_tz).dt.date
    else:
        dates_for_calendar = out[ts_col].dt.date

    # 3) Build a SAFE integer year range for the holiday objects
    years_series = pd.Series(dates_for_calendar)
    years_series = years_series.dropna().map(lambda d: int(pd.Timestamp(d).year))
    if years_series.empty:
        raise ValueError("No valid datetimes found to extract holiday years.")
    years = list(range(int(years_series.min()), int(years_series.max()) + 1))

    # 4) Construct holiday calendars
    qc_holidays = holidays.Canada(subdiv="QC", years=years, observed=observed)
    il_holidays = holidays.Israel(years=years, observed=observed)

   # 5) Flag membership
    out["is_qc_holiday"] = [ ("yes" if d in qc_holidays else "no") if pd.notna(pd.Timestamp(d)) else "no"
                             for d in dates_for_calendar ]
    out["is_jewish_holiday"] = [ ("yes" if d in il_holidays else "no") if pd.notna(pd.Timestamp(d)) else "no"
                                 for d in dates_for_calendar ]

    if include_names:
        out["qc_holiday_name"] = [ qc_holidays.get(d, "no") if pd.notna(pd.Timestamp(d)) else "no"
                                   for d in dates_for_calendar ]
        out["jewish_holiday_name"] = [ il_holidays.get(d, "no") if pd.notna(pd.Timestamp(d)) else "no"
                                       for d in dates_for_calendar ]

    return out

shift_types_dict = {'W1':'flow',
 'X1':'pod',
 'X3':'pod',
 'X4':'vertical',
 'X2':'vertical',
 'WOC1':'oncall',
 'WOC2':'oncall',
 'WOC3':'oncall',
 'X5':'pod',
 'W3':'overlap',
 'Y1':'pod',
 'Y3':'pod',
 'Y4':'vertical',
 'Y2':'vertical',
 'Y5':'pod',
 'Z1':'night',
 'Z2':'night',
 'D1':'pod',
 'R1':'pod',
 'P1':'vertical',
 'D2':'vertical',
 'OC1':'oncall',
 'OC2':'oncall',
 'V1':'flow',
 'A1':'pod',
 'G1':'vertical',
 'E1':'pod',
 'R2':'pod',
 'A2':'pod',
 'P2':'vertical',
 'E2':'vertical',
 'N1':'night',
 'N2':'night',
 'L2':'overlap',
 'L4':'overlap',
 'H1':'teaching',
 'B1':'vertical',
 'L1':'overlap',
 'W5':'overlap',
 'L6':'overlap',
 'B2':'vertical'}


# Load hourly data
df = pd.read_csv(
    'https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&raw=1')
df.ds = pd.to_datetime(df.ds, errors="coerce")
df['id'] = 'jgh'

# Load shift data
all_shifts_df = pd.read_csv('https://www.dropbox.com/scl/fi/yeyr2a7pj6nry8i2q3m0c/all_shifts.csv?rlkey=q1su2h8fqxfnlu7t1l2qe1w0q&raw=1')
all_shifts_df['shift_start'] = pd.to_datetime(all_shifts_df['shift_start']).dt.round('h')
all_shifts_df['shift_end'] = pd.to_datetime(all_shifts_df['shift_end']).dt.round('h')
all_shifts_df['shift_type'] = all_shifts_df['shift_short_name'].map(shift_types_dict)

# Create hourly rows
# We'll use a list comprehension to generate the range for each row
expanded_rows = []
for _, row in all_shifts_df.iterrows():
    # Create range. inclusive='left' means [start, end)
    # If start == end (e.g. 0 length shift after rounding), it will be empty, which is correct
    hours = pd.date_range(row['shift_start'], row['shift_end'], freq='h', inclusive='left')
    for h in hours:
        expanded_rows.append({
            'ds': h,
            'user': row['first_name']+row['last_name'],
            'shift_type': row['shift_type'],
            'shift_short_name': row['shift_short_name']
        })

expanded_df = pd.DataFrame(expanded_rows)

# Pivot
# index=timestamp, columns=user_id, values=shift_type
hourly_shifts_by_user_df = expanded_df.pivot_table(
    index='ds', 
    columns='user', 
    values='shift_type', 
    aggfunc='first' # In case of duplicates, take the first
)

# Fill NaNs
hourly_shifts_by_user_df = hourly_shifts_by_user_df.fillna('NotWorking')



ID_COL = "id"
TS_COL = "ds"
TARGETS = ['total_tbs']

df = df.copy()
df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")
df = df.dropna(subset=[TS_COL])

# Snap to exact hours (lowercase 'h' to avoid FutureWarning)
df[TS_COL] = df[TS_COL].dt.floor("h")

# Sort + dedupe
df = df.sort_values([ID_COL, TS_COL]).drop_duplicates(
    [ID_COL, TS_COL], keep="last")





# Call apply with include_groups=False if supported; else fall back
gb = df.groupby(ID_COL, group_keys=False)
try:
    df = gb.apply(regularize_hourly, include_groups=False)
except TypeError:
    # older pandas without include_groups
    df = gb.apply(regularize_hourly)

# Assert truly hourly (accept 'h' and 'H')
g = df[df[ID_COL] == "jgh"].sort_values(TS_COL)
freq = pd.infer_freq(g[TS_COL])
if not freq:
    raise ValueError("No inferable frequency after regularization.")
if to_offset(freq).name.lower() != "h":
    # extra check independent of infer_freq
    diffs = g[TS_COL].diff().dropna()
    bad = g.loc[diffs != pd.Timedelta(hours=1), TS_COL].head(10).tolist()
    raise ValueError(f"Non-1h gaps remain around: {bad}")

# Predict
print('Predicting basic forecast')
basic_forecast = pipeline.predict_df(
    df,
    prediction_length=24,
    # future_df = future_df.head(24),
    # quantile_levels=[0.1, 0.5, 0.9],
    # quantile_levels=[0.5],
    id_column=ID_COL,
    timestamp_column=TS_COL,
    target=TARGETS,
)

# basic_forecast


df_with_holidays = add_holiday_flags(df, ts_col='ds', include_names=True)

#create a dataframe with the next 24 hours timestamps hourly as column 'ds', with column 'id' jgh
future_df = hourly_shifts_by_user_df.reset_index()[hourly_shifts_by_user_df.reset_index()['ds'] > df['ds'].max()]
future_df['id'] = 'jgh'
future_df = add_holiday_flags(future_df, ts_col='ds', include_names=True)

# First, add holiday flags to future_df
future_df_with_added_holidays = add_holiday_flags(future_df, ts_col='ds', include_names=True)

# Then, select only the columns from future_df_with_added_holidays that are also in df_with_holidays
common_columns = [col for col in future_df_with_added_holidays.columns if col in df_with_holidays.columns]
future_df_with_holidays = future_df_with_added_holidays[common_columns]

# Predict
print('Predicting forecast with holidays')  
forecast_with_holidays = pipeline.predict_df(
    df_with_holidays,
    prediction_length=24,
    future_df = future_df_with_holidays.head(24),
    # quantile_levels=[0.1, 0.5, 0.9],
    # quantile_levels=[0.5],
    id_column=ID_COL,
    timestamp_column=TS_COL,
    target=TARGETS,
)

# forecast_with_holidays


df_with_staffing = df.merge(hourly_shifts_by_user_df, on='ds')
future_df_with_staffing = hourly_shifts_by_user_df.reset_index()[hourly_shifts_by_user_df.reset_index()['ds'] > df['ds'].max()]
future_df_with_staffing['id'] = 'jgh'

print('Predicting forecast with staffing')
forecast_with_staffing = pipeline.predict_df(
    df_with_staffing,
    prediction_length=24,
    future_df = future_df_with_staffing.head(24),
    # quantile_levels=[0.1, 0.5, 0.9],
    # quantile_levels=[0.5],
    id_column=ID_COL,
    timestamp_column=TS_COL,
    target=TARGETS,
)

# forecast_with_staffing

weather_df = pd.read_csv('https://www.dropbox.com/scl/fi/gmhwwld9z9yychg4r0yuk/weather.csv?rlkey=66c78m90aviamr0x0uu72pfr8&raw=1')
weather_df.ds = pd.to_datetime(weather_df.ds, errors="coerce")


future_weather_df = weather_df[weather_df.ds > df.ds.max()].head(24)
future_weather_df['id']='jgh'

print('Predicting forecast with weather')
# Predict
forecast_with_weather = pipeline.predict_df(
    #join df with weather_df on ds
    df.merge(weather_df, on='ds'),
    prediction_length=24,
    #weather_df where ds is greater than the max of df.ds.max()
    future_df = future_weather_df,
    # future_df = future_df.head(24),
    # quantile_levels=[0.1, 0.5, 0.9],
    quantile_levels=[0.5],
    id_column=ID_COL,
    timestamp_column=TS_COL,
    target=TARGETS,
)

#forecast_with_weather


# All variables forecast without future
# all_variable_df = add_holiday_flags(df_with_staffing, ts_col='ds', include_names=True).merge(weather_df, on='ds')
# print('Predicting all variables forecast without future')
# forecast_all_vars_without_future = pipeline.predict_df(
#     all_variable_df,
#     prediction_length=24,
#     # quantile_levels=[0.1, 0.5, 0.9],
#     quantile_levels=[0.5],
#     id_column=ID_COL,
#     timestamp_column=TS_COL,
#     target=TARGETS,
# )

#forecast_all_vars_without_future

# All variables forecast
print('Predicting all variables forecast')
all_variable_df = add_holiday_flags(df_with_staffing, ts_col='ds', include_names=True).merge(weather_df, on='ds')

forecast_all_vars_with_future = pipeline.predict_df(
    all_variable_df,
    prediction_length=24,
    #future_df should be future_df_with_staffing merged with future_weather_df on 'ds' and 'id'
    future_df = future_df_with_staffing.merge(future_weather_df, on=['ds', 'id']),
    # quantile_levels=[0.1, 0.5, 0.9],
    quantile_levels=[0.5],
    id_column=ID_COL,
    timestamp_column=TS_COL,
    target=TARGETS,
)   

#forecast_all_vars_with_future

#join the predictions columns of basic_forecast, forecast_with_holidays, forecast_with_staffing, forecast_with_weather, forecast_all_vars_without_future, forecast_all_vars_with_future on the 'ds' column
basic_forecast = basic_forecast[['ds','predictions']].rename(columns={'predictions':'basic_forecast'})
forecast_with_holidays = forecast_with_holidays[['ds','predictions']].rename(columns={'predictions':'forecast_with_holidays'})
forecast_with_staffing = forecast_with_staffing[['ds','predictions']].rename(columns={'predictions':'forecast_with_staffing'})
forecast_with_weather = forecast_with_weather[['ds','predictions']].rename(columns={'predictions':'forecast_with_weather'})
# forecast_all_vars_without_future = forecast_all_vars_without_future[['ds','predictions']].rename(columns={'predictions':'forecast_all_vars_without_future'})
forecast_all_vars_with_future = forecast_all_vars_with_future[['ds','predictions']].rename(columns={'predictions':'forecast_all_vars_with_future'})

pred_df = basic_forecast.merge(forecast_with_holidays, on='ds').merge(forecast_with_staffing, on='ds').merge(forecast_with_weather, on='ds').merge(forecast_all_vars_with_future, on='ds')
pred_df.head()




# df = df.merge(hourly_shifts_by_user_df, on='ds')
# df = add_holiday_flags(df, ts_col='ds', include_names=True)

# future_df = hourly_shifts_by_user_df.reset_index()[hourly_shifts_by_user_df.reset_index()['ds'] > df['ds'].max()]
# future_df['id'] = 'jgh'
# future_df = add_holiday_flags(future_df, ts_col='ds', include_names=True)


pred_df.to_csv('chronos_forecast.csv', index=False)

dropbox_app_key = os.environ.get("DROPBOX_APP_KEY")
dropbox_app_secret = os.environ.get("DROPBOX_APP_SECRET")
dropbox_refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")

# exchange the authorization code for an access token:
token_url = "https://api.dropboxapi.com/oauth2/token"
params = {
    "grant_type": "refresh_token",
    "refresh_token": dropbox_refresh_token,
    "client_id": dropbox_app_key,
    "client_secret": dropbox_app_secret
}
r = requests.post(token_url, data=params)

dropbox_access_token = r.json()['access_token']

dbx = dropbox.Dropbox(dropbox_access_token)

upload(dbx, 'chronos_forecast.csv', '', '',
            'chronos_forecast.csv', overwrite=True)
