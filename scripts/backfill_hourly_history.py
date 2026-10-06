"""Recover missing observations from every page of the rolling hourly PDF.

Dry run by default. Publish only with --publish-dropbox. Existing observations
are retained, and Dropbox revision checks prevent overwriting concurrent updates.
Dependencies: pandas, PyMuPDF, openpyxl; publishing also requires dropbox.
"""
import argparse
import io
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import fitz
import pandas as pd

from constants import column_names

REPORT_URL = ('https://www.dropbox.com/scl/fi/jbgh8weawfscab9emb70h/'
              'hourlyreport.pdf?rlkey=ai2jpkqhrhf3aqc0f9iaq1z0z&dl=1')
TARGETS = ['allData.csv', 'allData.xlsx',
           'allDataWithCalculatedColumns.csv', 'allDataWithCalculatedColumns.xlsx',
           'daily_inflow.csv', 'daily_inflow.xlsx']


def extract_report(path):
    records = []
    with fitz.open(path) as document:
        for page_number, page in enumerate(document, 1):
            words = page.get_text('words')
            has_dates = any(re.fullmatch(r'\d{2}/\d{2}/\d{2}', w[4]) for w in words)
            if not has_dates:
                continue  # The report includes alternate pages with only a footer.
            tables = page.find_tables().tables
            if len(tables) != 1 or tables[0].col_count != len(column_names):
                raise ValueError(f'Unexpected table layout on page {page_number}')
            table = tables[0]
            report_date = None
            for row_number, row in enumerate(table.extract()):
                if not row[1] or not row[1].isdigit():
                    continue
                if row[0]:
                    # Date cells sometimes include text from a merged row.
                    match = re.match(r'^(\d{2}/\d{2}/\d{2})(?:\s|$)', row[0])
                    if not match:
                        raise ValueError(f'Invalid date on page {page_number}')
                    report_date = pd.to_datetime(match[1], format='%m/%d/%y')
                hour = int(row[1])
                if report_date is None or not 1 <= hour <= 24:
                    raise ValueError(f'Invalid hour/date on page {page_number}')
                if any(not v or not re.fullmatch(r'\d+', v) for v in row[2:]):
                    raise ValueError(f'Missing/noninteger data on page {page_number}')
                numbers = [int(v) for v in row[2:]]
                # Independently check numeric cells against their PDF word positions.
                # This catches values assigned to the wrong table column.
                for j in range(1, len(row)):
                    x0, y0, x1, y1 = table.rows[row_number].cells[j]
                    cell_words = [w[4] for w in words
                                  if x0 <= (w[0] + w[2]) / 2 < x1
                                  and y0 <= (w[1] + w[3]) / 2 < y1]
                    if cell_words != [row[j]]:
                        raise ValueError(f'Word/table mismatch, page {page_number}, column {j}')
                if numbers[0] + numbers[2] != numbers[4]:
                    raise ValueError('Stretcher + ambulatory inflow does not equal total')
                if numbers[1] + numbers[3] != numbers[5]:
                    raise ValueError('Cumulative inflow components do not equal total')
                # Match get_current.py: report hour 24 belongs to next-day midnight.
                records.append([report_date + pd.Timedelta(hours=hour)] + numbers)
    frame = pd.DataFrame(records, columns=['ds'] + column_names[2:]).sort_values('ds')
    if frame.empty or frame.ds.duplicated().any():
        raise ValueError('Empty report or duplicate timestamps')
    expected = pd.date_range(frame.ds.min(), frame.ds.max(), freq='h')
    if len(expected.difference(frame.ds)):
        raise ValueError('Report has missing hours; refuse an incomplete recovery source')
    return frame


def merge_missing(history, report, start, end, expected_hours):
    columns = ['ds'] + column_names[2:]
    if list(history.columns) != columns:
        raise ValueError('Historical schema does not match the extractor')
    history = history.copy()
    history['ds'] = pd.to_datetime(history.ds)
    if history.ds.isna().any() or history.ds.duplicated().any():
        raise ValueError('History contains invalid or duplicate timestamps')
    wanted = pd.date_range(start, end, freq='h')
    if len(wanted) != expected_hours:
        raise ValueError('Expected hour count does not match the requested interval')
    if len(wanted.difference(report.ds)):
        raise ValueError('Requested hours are not all present in the PDF')
    old = history.set_index('ds')
    new = report.set_index('ds')
    overlap = old.index.intersection(new.index)
    differences = old.loc[overlap].ne(new.loc[overlap])
    # Report classifications can change retrospectively. Validate all the other
    # columns against history and never replace any existing record.
    revisable = ['INFLOW_STRETCHER', 'Infl_Stretcher_cum',
                 'INFLOW_AMBULATORY', 'Infl_Ambulatory_cum']
    if differences.drop(columns=revisable).any().any():
        raise ValueError('Report/history disagreement outside retrospective inflow classification')
    missing = wanted.difference(history.ds)
    added = new.loc[missing].rename_axis('ds').reset_index()
    merged = pd.concat([history, added], ignore_index=True).sort_values('ds').reset_index(drop=True)
    pd.testing.assert_frame_equal(merged.set_index('ds').loc[old.index], old, check_dtype=False)
    if merged.ds.duplicated().any() or len(wanted.difference(merged.ds)):
        raise ValueError('Recovery did not produce unique, complete hourly observations')
    audit = dict(added_hours=len(added), requested_hours=len(wanted),
                 interval_start=str(wanted.min()), interval_end=str(wanted.max()),
                 report_start=str(report.ds.min()), report_end=str(report.ds.max()),
                 history_rows_before=len(history), history_rows_after=len(merged),
                 unchanged_existing_rows=len(history), overlap_rows=len(overlap),
                 overlap_rows_matching_all_columns=int((~differences.any(axis=1)).sum()),
                 added_inflow=int(added.Inflow_Total.sum()))
    return merged, added, audit


def write_outputs(history, added, audit, output):
    output.mkdir(parents=True, exist_ok=True)
    calculated = history.copy()
    pod = ['TRG_HALLWAY_TBS', 'POD_GREEN_TBS', 'POD_YELLOW_TBS', 'POD_ORANGE_TBS']
    vertical = ['RAZ_TBS', 'AMBVERTTBS', 'QTrack_TBS', 'Garage_TBS']
    calculated['total_tbs'] = history[pod + vertical].sum(axis=1)
    calculated['vert_tbs'] = history[vertical].sum(axis=1)
    calculated['pod_tbs'] = history[pod].sum(axis=1)
    calculated['overflow'] = history.TRG_HALLWAY1 + history.POST_POD1
    daily = history.groupby(history.ds.dt.date).Inflow_Total.sum().reset_index(name='Daily_Inflow_Total')
    daily = daily.iloc[:-1]  # Preserve production's calendar-day convention.
    for name, frame in [('allData', history), ('allDataWithCalculatedColumns', calculated),
                        ('daily_inflow', daily)]:
        frame.to_csv(output / f'{name}.csv', index=False)
        frame.to_excel(output / f'{name}.xlsx', index_label='index')
    added.to_csv(output / 'recovered_hours.csv', index=False)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')


def publish(dbx, metadata, output, audit):
    from dropbox.files import WriteMode
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = '/backfill_backups'
    try:
        dbx.files_get_metadata(root)
    except Exception as error:
        # Only a confirmed missing path authorizes folder creation.
        import dropbox
        if not isinstance(error, dropbox.exceptions.ApiError) or not error.error.is_path() or not error.error.get_path().is_not_found():
            raise
        dbx.files_create_folder_v2(root)
    backup = f'{root}/{stamp}'
    dbx.files_create_folder_v2(backup)
    snapshots = {}
    for name in TARGETS:
        meta, response = dbx.files_download('/' + name)
        snapshots[name] = (meta, response.content)
        if name == 'allData.csv' and meta.rev != metadata.rev:
            raise RuntimeError('History changed during preparation; rerun using latest data')
        dbx.files_upload(response.content, f'{backup}/{name}', mode=WriteMode.add, mute=True)
    dbx.files_upload((output / 'audit.json').read_bytes(), f'{backup}/audit.json', mute=True)
    # Check every destination before the first write, and use conditional updates.
    for name, (meta, _) in snapshots.items():
        if dbx.files_get_metadata('/' + name).rev != meta.rev:
            raise RuntimeError(f'{name} changed during backup; no recovery outputs published')
    published = []
    try:
        for name, (meta, _) in snapshots.items():
            data = (output / name).read_bytes()
            dbx.files_upload(data, '/' + name, mode=WriteMode.update(meta.rev), mute=True)
            published.append(name)
            _, response = dbx.files_download('/' + name)
            if response.content != data:
                raise RuntimeError(f'Post-upload verification failed for {name}')
    except Exception:
        print(f'Publication stopped. Completed files: {published}. Originals at {backup}', flush=True)
        raise
    audit.update(published_files=published, backup_path=backup, verified=True)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    dbx.files_upload((output / 'audit.json').read_bytes(), f'{backup}/publication.json', mute=True)
    print(f'Published and verified {len(published)} outputs. Backup: {backup}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path)
    parser.add_argument('--history', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('backfill-output'))
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--expected-hours', type=int, required=True)
    parser.add_argument('--publish-dropbox', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dbx = metadata = None
    if args.publish_dropbox:
        if args.pdf or args.history:
            parser.error('Publishing always downloads current history and the live report')
        import dropbox
        dbx = dropbox.Dropbox(oauth2_refresh_token=os.environ['DROPBOX_REFRESH_TOKEN'],
                             app_key=os.environ['DROPBOX_APP_KEY'],
                             app_secret=os.environ['DROPBOX_APP_SECRET'], timeout=120)
        metadata, response = dbx.files_download('/allData.csv')
        history = pd.read_csv(io.BytesIO(response.content))
        args.pdf = args.output_dir / 'source-report.pdf'
        with urllib.request.urlopen(REPORT_URL, timeout=120) as response:
            args.pdf.write_bytes(response.read())
    else:
        if not args.pdf or not args.history:
            parser.error('Dry run requires --pdf and --history')
        history = pd.read_csv(args.history)
    report = extract_report(args.pdf)
    merged, added, audit = merge_missing(history, report, args.start, args.end, args.expected_hours)
    print(json.dumps(audit, indent=2), flush=True)
    write_outputs(merged, added, audit, args.output_dir)
    if dbx and len(added):
        publish(dbx, metadata, args.output_dir, audit)
    elif dbx:
        print('All requested hours already exist; no changes published.', flush=True)


if __name__ == '__main__':
    main()
