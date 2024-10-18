import os
import datetime
import requests
import camelot
import pandas as pd
# from deta import Deta
from dotenv import load_dotenv
import constants
from utils import upload
import dropbox

load_dotenv()

# deta = Deta(os.environ.get("DETA_PROJECT_KEY"))

# drive = deta.Drive("data")

URL = 'https://www.dropbox.com/s/ckijmipu33z3feg/HourlyReport.pdf?dl=1'
r = requests.get(URL, allow_redirects=True)
open('hourlyreport.pdf', 'wb').write(r.content)

tables = camelot.read_pdf(
    'hourlyreport.pdf',
    pages='1',
    flavor='stream',
    table_areas=['8,410,1000,50']
)

df = tables[0].df.reset_index(drop=True)
df.columns = constants.column_names
df.dateflg = df.dateflg[df.dateflg.str.strip() != '']
df.dateflg = df.dateflg.ffill()
for column in df.columns.tolist():
    if column in ['dateflg']:
        continue
    df[column] = df[column].astype('float').astype('int')
df["ds"] = pd.to_datetime(
    df["dateflg"] + " " + (df["timeflg"] - 1).astype(str) + ":00") + datetime.timedelta(hours=1)
df = df.set_index('ds').reset_index().drop(['dateflg', 'timeflg'], axis=1)

df.to_csv('current.csv', index=False)
df.to_excel('current.xlsx', index_label="index")

# drive.put(name='current.csv', path='current.csv')
# drive.put(name='current.xlsx', path='current.xlsx')


df = df.sort_values(by='ds', ascending=True)

# allData = pd.read_csv('https://drive.deta.sh/v1/b0x22rtxtdf/data/files/download?name=allData.csv',
#                       storage_options={'X-API-Key': os.environ.get("DETA_PROJECT_KEY")})
allData = pd.read_csv('https://www.dropbox.com/scl/fi/ksf0nbmmiort5khbrgr61/allData.csv?rlkey=75e735fjk4ifttjt553ukxt3k&dl=1')
allData.ds = pd.to_datetime(allData.ds)

allData = pd.concat([allData, df], ignore_index=True).drop_duplicates(subset='ds',
    keep='last').sort_values(by='ds', ascending=True)

allData.to_csv('allData.csv', index=False)
allData.to_excel('allData.xlsx', index_label="index")

# drive.put(name='allData.csv', path='allData.csv')
# drive.put(name='allData.xlsx', path='allData.xlsx')


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
    
    upload(dbx, 'hourlyreport.pdf', '', '',
               'hourlyreport.pdf', overwrite=True)
    
    upload(dbx, 'allData.csv', '', '',
               'allData.csv', overwrite=True)
    
    upload(dbx, 'allData.xlsx', '', '',
               'allData.xlsx', overwrite=True)
    
    upload(dbx, 'current.csv', '', '',
               'current.csv', overwrite=True)
    upload(dbx, 'current.xlsx', '', '',
               'current.xlsx', overwrite=True)
except:
    print('unable to upload to dropbox')

print(allData.tail(5))
