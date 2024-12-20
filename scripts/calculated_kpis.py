import base64
import json
import os
import requests
import pandas as pd
from prophet import Prophet
from tqdm import tqdm
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from utils import upload
import dropbox

load_dotenv()

allData = pd.read_csv(
    'https://www.dropbox.com/scl/fi/ksf0nbmmiort5khbrgr61/allData.csv?rlkey=75e735fjk4ifttjt553ukxt3k&dl=1')
allData.ds = pd.to_datetime(allData.ds)

df = allData.copy()
df.ds = pd.to_datetime(df.ds)

df['total_tbs'] = df[['TRG_HALLWAY_TBS',
                      'POD_GREEN_TBS',
                      'POD_YELLOW_TBS',
                      'POD_ORANGE_TBS',
                      'RAZ_TBS',
                      'AMBVERTTBS',
                      'QTrack_TBS',
                      'Garage_TBS']].sum(axis=1)
df['vert_tbs'] = df[[
    'RAZ_TBS',
    'AMBVERTTBS',
    'QTrack_TBS',
    'Garage_TBS']].sum(axis=1)
df['pod_tbs'] = df[['TRG_HALLWAY_TBS',
                    'POD_GREEN_TBS',
                    'POD_YELLOW_TBS',
                    'POD_ORANGE_TBS',
                    ]].sum(axis=1)
df.tail()

tbs_columns = ['total_tbs', 'vert_tbs', 'pod_tbs']

output = pd.DataFrame()
FIRST_RUN = True

for column in tqdm(tbs_columns):
    if column in ['ds']:
        continue
    try:
        print('working on '+column)

        m = Prophet(interval_width=0.95)
        m.fit(df[['ds', column]].rename(columns={column: 'y'}))
        future = m.make_future_dataframe(periods=24*1, freq='h')
        forecast = m.predict(future.tail(24*14))
        if FIRST_RUN:
            output['ds'] = forecast['ds']

        for forecast_column in ['yhat', 'yhat_lower', 'yhat_upper']:
            kwargs = {column+'_'+forecast_column: forecast[forecast_column]}
            output = output.assign(**kwargs)
        FIRST_RUN = False
    except:
        print(column + ' failed')

data = df.copy()
alerts = []
critical_alerts = []


def create_metric_graph(metric):
    # Calculate total patients to be seen as the sum of specified columns

    # Extract hour from the timestamp for grouping
    data['Hour'] = pd.to_datetime(data['ds']).dt.hour

    # Group by hour and calculate the average number of patients for each hour
    hourly_data = data.groupby('Hour')[metric].mean()

    # Identify the most recent timestamp's hour and its corresponding value
    most_recent_timestamp = pd.to_datetime(data['ds']).iloc[-1]
    most_recent_hour = most_recent_timestamp.hour
    most_recent_value = data.loc[pd.to_datetime(
        data['ds']) == most_recent_timestamp, metric].iloc[0]
    recent_data = data.loc[pd.to_datetime(data['ds']) >= pd.to_datetime(
        most_recent_timestamp.date()), metric]
    recent_data = recent_data.reset_index(drop=True)

    # Extract the day of the week for the most recent timestamp
    most_recent_day_of_week = most_recent_timestamp.dayofweek

    # Filter the data to include only rows matching the same day of the week
    same_day_data = data[pd.to_datetime(
        data['ds']).dt.dayofweek == most_recent_day_of_week]

    # Group by hour and calculate the average number of patients for this specific day of the week
    hourly_data_same_day = same_day_data.groupby('Hour')[metric].mean()

    prophet_data = output[['ds', metric+'_yhat',
                           metric+'_yhat_lower', metric+'_yhat_upper']].copy()
    prophet_data['Hour'] = pd.to_datetime(prophet_data['ds']).dt.hour
    prophet_data_today = prophet_data[prophet_data.ds.dt.date ==
                                      most_recent_timestamp.date()]
    prophet_data_today = prophet_data_today.reset_index(drop=True)

    prophet_data_today.tail()
    # Plot the updated graph with colors matching the example and no vertical grid lines
    plt.figure(figsize=(12, 6))
    # plt.bar(hourly_data_same_day.index, hourly_data_same_day, color='#5293ff', alpha=0.4, label='Expected')
    plt.bar(prophet_data_today.index,
            prophet_data_today[metric+'_yhat'], color='#5293ff', alpha=0.4, label='Expected')

    plt.bar(
        most_recent_hour, most_recent_value, color='#ff4d4d', alpha=0.4, label='Now'
    )
    plt.bar(
        recent_data.index, recent_data, color='#ff4d4d', alpha=0.2, label='Today'
    )
    plt.xlabel('Hour of the Day', fontsize=14)
    # plt.ylabel(metric, fontsize=14)
    if metric == 'total_tbs':
        plt.title('Total patients to be seen', fontsize=16)
    elif metric == 'vert_tbs':
        plt.title('Vertical patients to be seen', fontsize=16)
    elif metric == 'pod_tbs':
        plt.title('Pod patients to be seen', fontsize=16)
    else:
        plt.title(metric, fontsize=16)
    plt.xticks(range(0, 24), fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.grid(False, axis='x')  # Remove vertical grid lines
    plt.tight_layout()
    # plt.show()
    plt.savefig(metric+'.png')

    # Convert PNG to base64
    with open(metric+'.png', 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    if most_recent_value > prophet_data_today.iloc[most_recent_hour][metric+'_yhat_upper']:
        print(metric, 'critical')
        alerts.append({
            'metric': metric,
            'critical': True,
            'time': most_recent_timestamp,
            'value': most_recent_value,
            'forecast': prophet_data_today.iloc[most_recent_hour][metric+'_yhat_upper'],
            'forecast_time': prophet_data_today.iloc[most_recent_hour]['ds'],
            'forecast_value': prophet_data_today.iloc[most_recent_hour][metric+'_yhat'],
            'forecast_lower': prophet_data_today.iloc[most_recent_hour][metric+'_yhat_lower'],
            'forecast_upper': prophet_data_today.iloc[most_recent_hour][metric+'_yhat_upper'],
            'image': encoded_string
        })
        critical_alerts.append({
            'metric': metric,
            'critical': True,
            'time': most_recent_timestamp,
            'value': most_recent_value,
            'forecast': prophet_data_today.iloc[most_recent_hour][metric+'_yhat_upper'],
            'forecast_time': prophet_data_today.iloc[most_recent_hour]['ds'],
            'forecast_value': prophet_data_today.iloc[most_recent_hour][metric+'_yhat'],
            'forecast_lower': prophet_data_today.iloc[most_recent_hour][metric+'_yhat_lower'],
            'forecast_upper': prophet_data_today.iloc[most_recent_hour][metric+'_yhat_upper'],
            'image': encoded_string
        })
    else:
        print(metric, 'not critical')
        alerts.append({
            'metric': metric,
            'critical': False,
            'time': most_recent_timestamp,
            'value': most_recent_value,
            'forecast': prophet_data_today.iloc[most_recent_hour][metric+'_yhat_upper'],
            'forecast_time': prophet_data_today.iloc[most_recent_hour]['ds'],
            'forecast_value': prophet_data_today.iloc[most_recent_hour][metric+'_yhat'],
            'forecast_lower': prophet_data_today.iloc[most_recent_hour][metric+'_yhat_lower'],
            'forecast_upper': prophet_data_today.iloc[most_recent_hour][metric+'_yhat_upper'],
            'image': encoded_string
        })


for metric in tbs_columns:
    create_metric_graph(metric)

# print(alerts)

alerts_df = pd.DataFrame(alerts)
alerts_df.to_csv('calculated_KPIs_alerts.csv',
                 index=False)  # save alerts to csv
alerts_df.to_excel('calculated_KPIs_alerts.xlsx', index_label="index")

critical_alerts_df = pd.DataFrame(critical_alerts)
critical_alerts_df.to_csv(
    'calculated_KPIs_critical_alerts.csv', index=False)  # save alerts to csv
critical_alerts_df.to_excel(
    'calculated_KPIs_critical_alerts.xlsx', index_label="index")

dropbox_app_key = os.environ.get("DROPBOX_APP_KEY")
dropbox_app_secret = os.environ.get("DROPBOX_APP_SECRET")
dropbox_refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")

# exchange the authorization code for an access token:
TOKEN_URL = "https://api.dropboxapi.com/oauth2/token"
params = {
    "grant_type": "refresh_token",
    "refresh_token": dropbox_refresh_token,
    "client_id": dropbox_app_key,
    "client_secret": dropbox_app_secret
}
r = requests.post(TOKEN_URL, data=params, timeout=30)
# print(r.text)

dropbox_access_token = r.json()['access_token']

dbx = dropbox.Dropbox(dropbox_access_token)

for metric in tbs_columns:
    upload(dbx, metric+'.png', '', '',
           metric+'.png', overwrite=True)

upload(dbx, 'calculated_KPIs_alerts.csv', '', '',
            'calculated_KPIs_alerts.csv', overwrite=True)
upload(dbx, 'calculated_KPIs_alerts.xlsx', '', '',
            'calculated_KPIs_alerts.xlsx', overwrite=True)

upload(dbx, 'calculated_KPIs_critical_alerts.csv', '', '',
            'calculated_KPIs_critical_alerts.csv', overwrite=True)
upload(dbx, 'calculated_KPIs_critical_alerts.xlsx', '', '',
            'calculated_KPIs_critical_alerts.xlsx', overwrite=True)

# total_tbs_figure_as_base64 = base64.b64encode(
#     open('total_tbs.png', 'rb').read()).decode('utf-8')

# Check if total_tbs is critical by looking in critical_alerts_df
is_total_tbs_critical = False
if not critical_alerts_df.empty:
    is_total_tbs_critical = 'total_tbs' in critical_alerts_df['metric'].values

if is_total_tbs_critical:
    card_json = {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard", 
        "version": "1.5",
        "body": [
            {
                "type": "TextBlock",
                "text": "**CRITICAL FLOW ALERT**",
                "size": "Large",
                "weight": "Bolder",
                "color": "Attention"
            },
            {
                "type": "Image",
                "url": f"https://www.dropbox.com/scl/fi/9fye9nkzkxx78uh274ut4/total_tbs.png?rlkey=nnmup1ht4phevmt7yw9jyhav5&raw=1&timestamp={str(int(pd.Timestamp.now().timestamp()))}",
                "altText": "Daily Patient Volume Graph",
                "size": "Stretch"
            },
            {
                "type": "TextBlock",
                "text": "The current total number of patients to be seen is significantly above expected levels.",
                "wrap": True
            },
            {
                "type": "TextBlock",
                "text": "Consider mobilizing the on-call physician.",
                "wrap": True,
                "weight": "Bolder"
            },
            {
                "type": "ActionSet",
                "actions": [
                    {
                        "type": "Action.OpenUrl",
                        "title": "Go to Flow Dashboard",
                        "url": "https://app.powerbi.com/groups/me/reports/d22df078-20e6-4064-9c91-96d08d028897/ReportSectionbf2b1e80bc7570cb2ec4?experience=power-bi"
                    }
                ]
            }
        ]
    }
else:
    card_json = {}

# Convert to JSON string and write to file
with open('total_tbs_alert_adaptive_card.json', 'w') as f:
    json.dump(card_json, f, indent=2)

upload(dbx, 'total_tbs_alert_adaptive_card.json', '', '',
            'total_tbs_alert_adaptive_card.json', overwrite=True)
