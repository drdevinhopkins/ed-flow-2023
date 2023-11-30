import os
import datetime
import requests
import camelot
import pandas as pd
from deta import Deta
from dotenv import load_dotenv
import constants
from utils import upload
import dropbox

load_dotenv()

deta = Deta(os.environ.get("DETA_PROJECT_KEY"))

drive = deta.Drive("data")

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

drive.put(name='current.csv', path='current.csv')

df = df.sort_values(by='ds', ascending=True)

allData = pd.read_csv('https://drive.deta.sh/v1/b0x22rtxtdf/data/files/download?name=allData.csv',
                      storage_options={'X-API-Key': os.environ.get("DETA_PROJECT_KEY")})
allData.ds = pd.to_datetime(allData.ds)

allData = pd.concat([allData, df], ignore_index=True).drop_duplicates(
    keep='last').sort_values(by='ds', ascending=True)

allData.to_csv('allData.csv', index=False)

drive.put(name='allData.csv', path='allData.csv')

dropbox_access_token = os.environ.get("DROPBOX_ACCESS_TOKEN")
print(dropbox_access_token)
dbx = dropbox.Dropbox(oauth2_access_token=dropbox_access_token)

upload(dbx, 'hourlyreport.pdf', '', '',
           'hourlyreport.pdf', overwrite=True)

upload(dbx, 'allData.csv', '', '',
           'allData.csv', overwrite=True)

print(allData.tail(5))
