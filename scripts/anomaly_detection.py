import os
import pandas as pd
from prophet import Prophet
from tqdm import tqdm
from deta import Deta
from dotenv import load_dotenv
from utils import upload
import dropbox

load_dotenv()

deta = Deta(os.environ.get("DETA_PROJECT_KEY"))

drive = deta.Drive("data")


allData = pd.read_csv('https://drive.deta.sh/v1/b0x22rtxtdf/data/files/download?name=allData.csv',
                      storage_options={'X-API-Key': os.environ.get("DETA_PROJECT_KEY")})
allData.ds = pd.to_datetime(allData.ds)
print(allData.tail(1))

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

drive.put(name='anomaly_detection_ranges.csv',
          path='anomaly_detection_ranges.csv')

dropbox_access_token = os.environ.get("DROPBOX_ACCESS_TOKEN")
dbx = dropbox.Dropbox(dropbox_access_token)

upload(dbx, 'anomaly_detection_ranges.csv', '', '',
           'anomaly_detection_ranges.csv', overwrite=True)

print(allData.tail(5))
