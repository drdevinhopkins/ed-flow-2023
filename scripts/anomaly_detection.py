import os
import requests
import pandas as pd
import numpy as np
np.float_ = np.float64
from prophet import Prophet
from tqdm import tqdm
# from deta import Deta
from dotenv import load_dotenv
from utils import upload
import dropbox

load_dotenv()

allData = pd.read_csv('https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&dl=1')

allData.ds = pd.to_datetime(allData.ds)
print(allData.tail(1))
print('length of allData: '+str(len(allData)))

df = allData.copy()
df.isna().sum().sum()
df.ds = pd.to_datetime(df.ds)

output = pd.DataFrame()
FIRST_RUN = True

for column in tqdm(df.columns.to_list()):
    if column in ['ds']:
        continue
    try:
        print('working on '+column)

        m = Prophet(interval_width=0.95)
        m.fit(df[['ds', column]].rename(columns={column: 'y'}))
        future = m.make_future_dataframe(periods=24*1, freq='H')
        forecast = m.predict(future.tail(24*14))
        if FIRST_RUN:
            output['ds'] = forecast['ds']

        for forecast_column in ['yhat', 'yhat_lower', 'yhat_upper']:
            kwargs = {column+'_'+forecast_column: forecast[forecast_column]}
            output = output.assign(**kwargs)
        FIRST_RUN = False
    except:
        print(column + ' failed')

output.to_csv('anomaly_detection_ranges.csv', index=False)
output.to_excel('anomaly_detection_ranges.xlsx', index_label="index")

try:
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
    
    upload(dbx, 'anomaly_detection_ranges.csv', '', '',
               'anomaly_detection_ranges.csv', overwrite=True)
    upload(dbx, 'anomaly_detection_ranges.xlsx', '', '',
               'anomaly_detection_ranges.xlsx', overwrite=True)
except:
    print('unable to upload to dropbox')

print(allData.tail(5))
