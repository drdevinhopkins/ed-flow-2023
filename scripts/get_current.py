import os
import datetime
import requests
import camelot
import pandas as pd
from deta import Deta
from dotenv import load_dotenv
import constants

load_dotenv()

deta = Deta(os.environ.get("DETA_PROJECT_KEY"))

data = deta.Drive("data")

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
    df["dateflg"] + " " + (df["timeflg"] - 1).astype(str) + ":00", format='mixed') + datetime.timedelta(hours=1)
df = df.set_index('ds').reset_index().drop(['dateflg', 'timeflg'], axis=1)

df.to_csv('current.csv', index=False)

data.put(name='current.csv', path='current.csv')

print(df.head())