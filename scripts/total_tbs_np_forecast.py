import os
import tempfile
import pandas as pd
from dotenv import load_dotenv
import dropbox
import requests
from utils import upload
from neuralprophet import NeuralProphet
from neuralprophet import load

load_dotenv()


def reformat_forecast(forecast):
    forecast_data = []

    for i in range(24):
        step_col = f"step{i}"
        quantile_20_col = f"step{i} 20.0%"  # Column name for 20th percentile
        quantile_80_col = f"step{i} 80.0%"  # Column name for 80th percentile

        forecast_data.append({
            "ds": forecast["ds"].iloc[-1] + pd.Timedelta(hours=i),
            "yhat": forecast[step_col].iloc[-1],
            # Add lower bound (20th percentile)
            "yhat_lower": forecast[quantile_20_col].iloc[-1],
            # Add upper bound (80th percentile)
            "yhat_upper": forecast[quantile_80_col].iloc[-1]
        })

    new_forecast_df = pd.DataFrame(forecast_data)
    return new_forecast_df


def fetch_model_to_tmp(url: str, token: str | None = None) -> str:
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        fd, tmp_path = tempfile.mkstemp(suffix=".np")
        with os.fdopen(fd, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return tmp_path


MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
ASSET = f"inflow_total_np-{MODEL_VERSION}.np"
URL = f"https://github.com/drdevinhopkins/ed-flow-2023/releases/download/v{MODEL_VERSION}/{ASSET}"
TOKEN = os.getenv("GITHUB_TOKEN")  # required if private

local_path = fetch_model_to_tmp(URL, TOKEN)
inflow_total_np_m = load(local_path)

MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
ASSET = f"total_tbs-{MODEL_VERSION}.np"
URL = f"https://github.com/drdevinhopkins/ed-flow-2023/releases/download/v{MODEL_VERSION}/{ASSET}"
TOKEN = os.getenv("GITHUB_TOKEN")  # required if private

local_path = fetch_model_to_tmp(URL, TOKEN)
total_tbs_m = load(local_path)

data = pd.read_csv(
    'https://www.dropbox.com/scl/fi/ksf0nbmmiort5khbrgr61/allData.csv?rlkey=75e735fjk4ifttjt553ukxt3k&raw=1')
data.ds = pd.to_datetime(data.ds)
data = data.sort_values('ds')

df = data.copy()
df = df[['ds', 'Inflow_Total']].rename(
    columns={'ds': 'ds', 'Inflow_Total': 'y'})
df.tail()
df_future = inflow_total_np_m.make_future_dataframe(
    df[['ds', 'y']], periods=24)

if hasattr(inflow_total_np_m, "trainer") and inflow_total_np_m.trainer is not None:
    # Disable the built-in TensorBoard logger
    inflow_total_np_m.trainer.logger = False
    # Also disable callbacks that might try to log
    inflow_total_np_m.trainer.callbacks = []

forecast = inflow_total_np_m.predict(df_future, decompose=False, raw=True)
output_df = reformat_forecast(forecast)
inflow_total_np = output_df.copy()
# inflow_total_np

data['y'] = data['POD_GREEN_TBS']+data['POD_YELLOW_TBS']+data['POD_ORANGE_TBS'] + \
    data['TRG_HALLWAY_TBS']+data['RAZ_TBS'] + \
    data['AMBVERTTBS']+data['QTrack_TBS']+data['Garage_TBS']
df = data.copy()
df = df[['ds', 'y', 'Inflow_Total']]


df_future = total_tbs_m.make_future_dataframe(df[['ds', 'y', 'Inflow_Total']], periods=24, regressors_df=inflow_total_np[[
    'ds', 'yhat']].rename(columns={'ds': 'ds', 'yhat': 'Inflow_Total'}))

if hasattr(total_tbs_m, "trainer") and total_tbs_m.trainer is not None:
    # Disable the built-in TensorBoard logger
    total_tbs_m.trainer.logger = False
    # Also disable callbacks that might try to log
    total_tbs_m.trainer.callbacks = []

forecast = total_tbs_m.predict(df_future, decompose=False, raw=True)

output_df = reformat_forecast(forecast)

output_df.to_csv('total_tbs_np.csv', index=False)

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
# print(r.text)

dropbox_access_token = r.json()['access_token']

dbx = dropbox.Dropbox(dropbox_access_token)

upload(dbx, 'total_tbs_np.csv', '', '',
            'total_tbs_np.csv', overwrite=True)

print(output_df)
