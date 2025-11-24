from chronos import BaseChronosPipeline, Chronos2Pipeline
import pandas as pd
import os
import pandas as pd
from dotenv import load_dotenv
import requests
from utils import upload
import dropbox
import pandas as pd
from pandas.tseries.frequencies import to_offset

load_dotenv()

# Load the Chronos-2 pipeline
# GPU recommended for faster inference, but CPU is also supported
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    # device_map="cuda"
    device_map="cpu"
)

df = pd.read_csv(
    'https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&raw=1')
df.ds = pd.to_datetime(df.ds, errors="coerce")
df['id'] = 'jgh'


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
pred_df = pipeline.predict_df(
    df,
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=ID_COL,
    timestamp_column=TS_COL,
    target=TARGETS,
)

pred_df.to_csv('chronos_forecast.csv', index=False)
pred_df.to_excel('chronos_forecast.xlsx', index_label="index")

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
