import os
import requests
import pandas as pd
# from deta import Deta
from dotenv import load_dotenv
from utils import upload
import dropbox

load_dotenv()

anomaly_detection_ranges = pd.read_csv(
    'https://www.dropbox.com/scl/fi/fjz0am427gw35sz7l994m/anomaly_detection_ranges.csv?rlkey=lib9w0jz2zei5n566jv76o7ol&dl=1')

anomaly_detection_ranges.ds = pd.to_datetime(anomaly_detection_ranges.ds)

current_df = pd.read_csv(
    'https://www.dropbox.com/scl/fi/4zmm1bpwtrhrqtvce8aqq/current.csv?rlkey=1wnfrlal5jefwkx4fg7jxl20o&dl=1')

current_df.ds = pd.to_datetime(current_df.ds)

current_df['total_tbs'] = current_df[['TRG_HALLWAY_TBS',
                                      'POD_GREEN_TBS',
                                      'POD_YELLOW_TBS',
                                      'POD_ORANGE_TBS',
                                      'RAZ_TBS',
                                      'AMBVERTTBS',
                                      'QTrack_TBS',
                                      'Garage_TBS']].sum(axis=1)
current_df['vert_tbs'] = current_df[[
    'RAZ_TBS',
    'AMBVERTTBS',
    'QTrack_TBS',
    'Garage_TBS']].sum(axis=1)
current_df['pod_tbs'] = current_df[['TRG_HALLWAY_TBS',
                                    'POD_GREEN_TBS',
                                    'POD_YELLOW_TBS',
                                    'POD_ORANGE_TBS',
                                    ]].sum(axis=1)

current_df['overflow'] = current_df['TRG_HALLWAY1']+current_df['POST_POD1']

current = current_df.head(1).iloc[0]

anomaly_detection_ranges = anomaly_detection_ranges.set_index(
    'ds').loc[current.ds]

alert_types = [
    'INFLOW_STRETCHER',
    # 'Infl_Stretcher_cum',
    'INFLOW_AMBULATORY',
    # 'Infl_Ambulatory_cum',
    'Inflow_Total',
    # 'Inflow_Cum_Total',
    'INFLOW_AMBULANCES',
    # 'Infl_Ambulances_cum',
    # 'FLS',
    # 'CUM_ADMREQ',
    # 'CUM_BA1',
    # 'WAITINGADM',
    # 'TTStr',
    'TRG_HALLWAY1',
    'TRG_HALLWAY_TBS',
    # 'reoriented_cum',
    # 'reoriented_cum_MD',
    'QTRACK1',
    # 'RESUS',
    # 'Pod_T',
    # 'POD_GREEN',
    'POD_GREEN_TBS',
    # 'POD_YELLOW',
    'POD_YELLOW_TBS',
    # 'POD_ORANGE',
    'POD_ORANGE_TBS',
    'POD_CONS_MORE2H',
    'POD_IMCONS_MORE4H',
    'POD_XRAY_MORE2H',
    'POD_CT_MORE2H',
    # 'POST_POD1',
    # 'VERTSTRET',
    'RAZ_TBS',
    # 'RAZ_LAZYBOY',
    # 'RAZ_WAITINGREZ',
    # 'AMBVERT1',
    'AMBVERTTBS',
    'QTrack_TBS',
    # 'Garage_TBS',
    'RAZ_CONS_MORE2H',
    'RAZ_IMCONS_MORE4H',
    'RAZ_XRAY_MORE2H',
    'RAZ_CT_MORE2H1',
    # 'PSYCH1',
    # 'PSYCH_WAITINGADM'
]

alerts = []

for column in alert_types:
    try:
        if current[column] > anomaly_detection_ranges[column+'_yhat_upper']:
            alerts.append({'metric': column, 'value': current[column], 'yhat_upper': round(
                anomaly_detection_ranges[column+'_yhat_upper'], 1), 'ds': current.ds})
    except:
        continue

alerts_df = pd.DataFrame(alerts)

alerts_df.to_csv('alerts.csv', index=False)
alerts_df.to_excel('alerts.xlsx', index_label="index")


# drive.put(name='alerts.csv',
#           path='alerts.csv')
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

    upload(dbx, 'alerts.csv', '', '',
           'alerts.csv', overwrite=True)
    upload(dbx, 'alerts.xlsx', '', '',
           'alerts.xlsx', overwrite=True)
except:
    print('unable to upload to dropbox')

print(alerts_df)
