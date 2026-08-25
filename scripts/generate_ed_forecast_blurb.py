#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from datetime import datetime
from statistics import median
from zoneinfo import ZoneInfo

import dropbox
import pandas as pd
import requests
from dropbox.exceptions import ApiError
from dropbox.files import WriteMode

TZ = ZoneInfo('America/Montreal')
RUN_HOURS = {7, 11, 15, 19}
STRETCHER_CAPACITY = 53

FORECAST_PATH = os.getenv('DROPBOX_FORECAST_V21_PATH', '/forecast-v2.1.csv')
CURRENT_PATH = os.getenv('DROPBOX_CURRENT_PATH', '/current.csv')
ONCALL_PROB_PATH = os.getenv('DROPBOX_ONCALL_PROBABILITY_PATH', '/oncall_need_probability.csv')
ONCALL_IMPACT_PATH = os.getenv('DROPBOX_ONCALL_IMPACT_PATH', '/oncall_impact_summary.csv')
BLURB_PATH = os.getenv('DROPBOX_BLURB_PATH', '/hourly_forecast_blurbs.csv')

COLUMNS = [
    'generated_at_local',
    'forecast_data_time_local',
    'blurb',
    'oncall_recommendation',
    'oncall_rationale',
    'source_status',
]


def num(value):
    value = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
    return None if pd.isna(value) else float(value)


def fmt(value):
    if value is None:
        return 'unknown'
    return str(int(round(value))) if abs(value - round(value)) < 0.05 else f'{value:.1f}'


def fmt_time(ts):
    text = pd.Timestamp(ts).strftime('%I %p')
    return text[1:] if text.startswith('0') else text


def sum_cols(row, cols):
    values = [num(row.get(c)) for c in cols]
    return None if any(v is None for v in values) else float(sum(values))


def is_not_found(err):
    try:
        return err.error.is_path() and err.error.get_path().is_not_found()
    except Exception:
        return False


def is_conflict(err):
    try:
        return err.error.is_path() and err.error.get_path().is_conflict()
    except Exception:
        return False


def get_dbx():
    key = os.environ.get('DROPBOX_APP_KEY')
    secret = os.environ.get('DROPBOX_APP_SECRET')
    refresh = os.environ.get('DROPBOX_REFRESH_TOKEN')
    if not all([key, secret, refresh]):
        raise RuntimeError('DROPBOX_APP_KEY, DROPBOX_APP_SECRET and DROPBOX_REFRESH_TOKEN are required')
    response = requests.post(
        'https://api.dropboxapi.com/oauth2/token',
        data={
            'grant_type': 'refresh_token',
            'refresh_token': refresh,
            'client_id': key,
            'client_secret': secret,
        },
        timeout=30,
    )
    response.raise_for_status()
    return dropbox.Dropbox(response.json()['access_token'], timeout=60)


def download_csv(dbx, path, required=True):
    try:
        metadata, response = dbx.files_download(path)
    except ApiError as err:
        if not required and is_not_found(err):
            return pd.DataFrame(), None
        raise
    if not response.content.strip():
        return pd.DataFrame(), metadata
    return pd.read_csv(io.BytesIO(response.content)), metadata


def prep_forecast(frame):
    frame = frame.copy()
    frame['ds'] = pd.to_datetime(frame['ds'], errors='coerce')
    if 'forecast_origin' in frame:
        frame['forecast_origin'] = pd.to_datetime(frame['forecast_origin'], errors='coerce')
    for col in [
        'forecast', 'anomaly_yhat', 'anomaly_yhat_lower', 'anomaly_yhat_upper',
        'feature_effect', 'feature_effect_pct',
    ]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
    return frame.dropna(subset=['ds'])


def prep_current(frame):
    frame = frame.copy()
    frame['ds'] = pd.to_datetime(frame['ds'], errors='coerce')
    return frame.dropna(subset=['ds']).sort_values('ds')


def latest_metrics(current):
    row = current.iloc[-1]
    ds = pd.Timestamp(row['ds'])
    pod = sum_cols(row, ['TRG_HALLWAY_TBS', 'POD_GREEN_TBS', 'POD_YELLOW_TBS', 'POD_ORANGE_TBS'])
    vertical = sum_cols(row, ['RAZ_TBS', 'AMBVERTTBS', 'QTrack_TBS', 'Garage_TBS'])
    total = None if pod is None or vertical is None else pod + vertical
    room = num(row.get('POST_POD1'))
    prepod = num(row.get('TRG_HALLWAY1'))
    overflow = None if room is None or prepod is None else room + prepod
    metrics = {
        'Overflow': overflow,
        'POD_TBS': pod,
        'TTStr': num(row.get('TTStr')),
        'Total_TBS': total,
        'Vertical_TBS': vertical,
        'WAITINGADM': num(row.get('WAITINGADM')),
    }
    return ds, metrics, room, prepod


def anomaly_text(forecast, ds, metrics):
    bits = []
    for target, actual in metrics.items():
        if actual is None:
            continue
        rows = forecast[(forecast['target_name'] == target) & (forecast['ds'] == ds)]
        if rows.empty:
            continue
        row = rows.iloc[-1]
        lo, hi = num(row.get('anomaly_yhat_lower')), num(row.get('anomaly_yhat_upper'))
        if hi is not None and actual > hi:
            if target == 'TTStr':
                bits.append(f'{target} {actual / STRETCHER_CAPACITY * 100:.0f}% occupancy (upper expected bound ~{hi / STRETCHER_CAPACITY * 100:.0f}%)')
            else:
                bits.append(f'{target} {fmt(actual)} (upper expected bound ~{fmt(hi)})')
        elif lo is not None and actual < lo:
            if target == 'TTStr':
                bits.append(f'{target} {actual / STRETCHER_CAPACITY * 100:.0f}% occupancy (lower expected bound ~{lo / STRETCHER_CAPACITY * 100:.0f}%)')
            else:
                bits.append(f'{target} {fmt(actual)} (lower expected bound ~{fmt(lo)})')
    if not bits:
        return ''
    label = 'Current significant anomaly' if len(bits) == 1 else 'Current significant anomalies'
    return f"{label}: {'; '.join(bits)}."


def overflow_text(overflow, room, prepod):
    if overflow is None:
        return ''
    detail = '' if room is None or prepod is None else f' ({fmt(room)} in overflow-room space and {fmt(prepod)} in prepod)'
    if overflow <= 16:
        return f'Overflow is {fmt(overflow)}{detail}, within the range usually handled comfortably by the first two overflow rooms.'
    if overflow < 30:
        return f'Overflow is {fmt(overflow)}{detail}, beyond the comfortably usable first two rooms; prepod accumulation increasingly depends on how quickly rooms 3–5 can be staffed and opened.'
    if overflow <= 40:
        return f'Overflow is {fmt(overflow)}{detail}, in the range that generally requires most or all overflow rooms; prepod accumulation is likely if rooms 3–5 are not fully staffed and opened promptly.'
    return f'Overflow is {fmt(overflow)}{detail}, beyond nominal five-room overflow capacity and consistent with substantial prepod/triage-hallway pressure.'


def future_window(forecast, ds, hours=12):
    future = forecast[forecast['row_type'].astype(str).eq('forecast')]
    window = future[(future['ds'] > ds) & (future['ds'] <= ds + pd.Timedelta(hours=hours))]
    if window.empty:
        window = future[(future['ds'] > ds) & (future['ds'] <= ds + pd.Timedelta(hours=24))]
    return window.copy()


def target_rows(window, target):
    return window[window['target_name'].eq(target)].dropna(subset=['forecast']).sort_values('ds')


def trajectory_text(window, metrics):
    if window.empty:
        return 'No fresh 12–24 hour forecast trajectory is available.'
    clauses = []
    tt = target_rows(window, 'TTStr')
    ov = target_rows(window, 'Overflow')
    wa = target_rows(window, 'WAITINGADM')

    if not tt.empty:
        peak = tt.loc[tt['forecast'].idxmax()]
        end = tt.iloc[-1]
        peak_pct = float(peak['forecast']) / STRETCHER_CAPACITY * 100
        end_pct = float(end['forecast']) / STRETCHER_CAPACITY * 100
        current = metrics.get('TTStr')
        current_pct = None if current is None else current / STRETCHER_CAPACITY * 100
        if current_pct is not None and peak_pct > current_pct + 5:
            clauses.append(f'stretcher occupancy builds from ~{current_pct:.0f}% to ~{peak_pct:.0f}% around {fmt_time(peak["ds"])}')
        elif current_pct is not None and end_pct < current_pct - 5:
            clauses.append(f'stretcher occupancy eases from ~{current_pct:.0f}% toward ~{end_pct:.0f}% by {fmt_time(end["ds"])}')
        else:
            clauses.append(f'stretcher occupancy is ~{end_pct:.0f}% by {fmt_time(end["ds"])}')

    if not ov.empty:
        peak = ov.loc[ov['forecast'].idxmax()]
        end = ov.iloc[-1]
        current = metrics.get('Overflow')
        peak_v, end_v = float(peak['forecast']), float(end['forecast'])
        if current is not None and peak_v > current + 2:
            clauses.append(f'Overflow peaks near {fmt(peak_v)} around {fmt_time(peak["ds"])}')
        elif current is not None and end_v < current - 2:
            clauses.append(f'Overflow eases toward ~{fmt(end_v)} by {fmt_time(end["ds"])}')
        else:
            clauses.append(f'Overflow is ~{fmt(end_v)} by {fmt_time(end["ds"])}')
        if 'forecast_anomaly' in ov:
            flagged = ov[ov['forecast_anomaly'].astype(str).str.lower().eq('yes')]
            if not flagged.empty:
                clauses.append(f'Overflow remains forecast-anomalous through at least {fmt_time(flagged.iloc[-1]["ds"])}')

    current_wa = metrics.get('WAITINGADM')
    if current_wa is not None and not wa.empty:
        end_v = float(wa.iloc[-1]['forecast'])
        if end_v <= current_wa - 3:
            clauses.append(f'boarding eases from {fmt(current_wa)} waiting for admission to ~{fmt(end_v)}')
        elif end_v >= current_wa + 3:
            clauses.append(f'boarding rises from {fmt(current_wa)} waiting for admission to ~{fmt(end_v)}')

    if not clauses:
        return 'The near-term forecast is broadly stable.'
    if len(clauses) == 1:
        return f'The model expects {clauses[0]}.'
    return f"The model expects {', '.join(clauses[:-1])}, and {clauses[-1]}."


STAFF_THRESH = {'TTStr': 2.0, 'Total_TBS': 1.0, 'Vertical_TBS': 1.0, 'POD_TBS': 0.75, 'Overflow': 1.0, 'WAITINGADM': 1.0}


def staffing_text(window):
    if window.empty or 'feature_family' not in window or 'feature_effect' not in window:
        return ''
    rows = window[window['feature_family'].astype(str).str.lower().eq('staffing')]
    meaningful = []
    for target, threshold in STAFF_THRESH.items():
        values = pd.to_numeric(rows.loc[rows['target_name'].eq(target), 'feature_effect'], errors='coerce').dropna()
        if values.empty:
            continue
        med = float(values.median())
        peak = float(values.loc[values.abs().idxmax()])
        representative = med if abs(med) >= threshold else peak
        sign = 1 if representative > 0 else -1
        consistency = float((values * sign > 0).mean())
        if abs(representative) >= threshold and consistency >= 0.60:
            meaningful.append((target, representative, representative / threshold))
    if len(meaningful) < 2:
        return ''
    overall = float(median([x[2] for x in meaningful]))
    same = [x for x in meaningful if x[2] * overall > 0]
    if abs(overall) < 0.5 or len(same) < 2:
        return ''
    direction = 'weaker than usual' if overall > 0 else 'stronger than usual'
    verb = 'adding' if overall > 0 else 'reducing'
    examples = sorted(same, key=lambda x: abs(x[2]), reverse=True)[:2]
    labels = []
    for target, effect, _ in examples:
        value = abs(effect)
        if target == 'TTStr':
            labels.append(f'~{fmt(value)} stretcher patients')
        elif target == 'WAITINGADM':
            labels.append(f'~{fmt(value)} waiting-admission patients')
        elif target == 'Overflow':
            labels.append(f'~{fmt(value)} overflow patients')
        else:
            labels.append(f'~{fmt(value)} {target.replace("_", " ")}')
    return f"Today's modeled physician/staffing mix appears {direction} from a flow standpoint, with staffing-associated effects {verb} {' and '.join(labels)} at the more affected horizons; this is associative, not causal."


IMPACT_THRESH = {'total_tbs': 1.0, 'pod_tbs': 0.75, 'vertical_tbs': 1.0, 'stretcher_occupancy': 3.0, 'overflow': 1.0}


def oncall_assessment(prob, impact, now_naive):
    if prob.empty:
        return 'NO CLEAR RECOMMENDATION', 'on-call probability output is unavailable', None
    prob = prob.copy()
    prob['ds'] = pd.to_datetime(prob['ds'], errors='coerce')
    prob['calibrated_probability'] = pd.to_numeric(prob['calibrated_probability'], errors='coerce')
    latest_ds = prob['ds'].max()
    latest = prob[prob['ds'].eq(latest_ds)].dropna(subset=['calibrated_probability'])
    if latest.empty:
        return 'NO CLEAR RECOMMENDATION', 'calibrated on-call probabilities are unavailable', latest_ds
    pmax = float(latest['calibrated_probability'].max())
    age = (now_naive - pd.Timestamp(latest_ds)).total_seconds() / 3600
    horizons = sorted(int(x) for x in pd.to_numeric(latest['horizon_hours'], errors='coerce').dropna().unique())
    horizon_text = '/'.join(map(str, horizons)) + 'h' if horizons else 'available horizons'
    if age > 4:
        return 'NO CLEAR RECOMMENDATION', f'on-call model is stale ({age:.1f}h old); max calibrated need was {pmax * 100:.1f}%', latest_ds
    if impact.empty:
        rec = 'NOT INDICATED' if pmax < 0.05 else 'NO CLEAR RECOMMENDATION'
        return rec, f'max calibrated need {pmax * 100:.1f}% over {horizon_text}; impact output unavailable', latest_ds

    impact = impact.copy()
    impact['estimated_improvement'] = pd.to_numeric(impact['estimated_improvement'], errors='coerce')
    impact = impact[impact['target_name'].isin(IMPACT_THRESH)].dropna(subset=['estimated_improvement'])
    if impact.empty:
        return 'NO CLEAR RECOMMENDATION', f'max calibrated need {pmax * 100:.1f}% over {horizon_text}; usable impact comparisons unavailable', latest_ds

    positive_fraction = float((impact['estimated_improvement'] > 0).mean())
    meaningful_pos = 0
    meaningful_neg = 0
    for target, threshold in IMPACT_THRESH.items():
        values = impact.loc[impact['target_name'].eq(target), 'estimated_improvement']
        if values.empty:
            continue
        meaningful_pos += int(float(values.max()) >= threshold)
        meaningful_neg += int(float(values.min()) <= -threshold)
    benefit = positive_fraction >= 0.60 and meaningful_pos >= 2
    no_benefit = positive_fraction <= 0.50 or meaningful_neg >= 2

    if pmax >= 0.50 and benefit:
        rec = 'USE'
    elif pmax >= 0.20 and benefit:
        rec = 'CONSIDER'
    elif pmax < 0.10 and no_benefit:
        rec = 'NOT INDICATED'
    elif pmax < 0.05:
        rec = 'NOT INDICATED'
    else:
        rec = 'NO CLEAR RECOMMENDATION'

    effect_text = 'modeled activation shows a consistent meaningful flow benefit' if benefit else ('modeled activation shows no consistent meaningful flow benefit' if no_benefit else 'modeled activation effect is mixed or small')
    return rec, f'max calibrated need {pmax * 100:.1f}% over {horizon_text}; {effect_text}', latest_ds


def freshness(now_naive, current_ds, forecast, oncall_ds):
    origins = forecast['forecast_origin'].dropna() if 'forecast_origin' in forecast else pd.Series(dtype='datetime64[ns]')
    origin = pd.Timestamp(origins.max()) if not origins.empty else None
    current_age = (now_naive - current_ds).total_seconds() / 3600
    forecast_age = None if origin is None else (now_naive - origin).total_seconds() / 3600
    oncall_age = None if oncall_ds is None else (now_naive - pd.Timestamp(oncall_ds)).total_seconds() / 3600
    stale = []
    if current_age > 2:
        stale.append(f'current {current_age:.1f}h old')
    if forecast_age is None:
        stale.append('forecast origin unknown')
    elif forecast_age > 4:
        stale.append(f'forecast {forecast_age:.1f}h old')
    if oncall_age is None:
        stale.append('on-call unavailable')
    elif oncall_age > 4:
        stale.append(f'on-call {oncall_age:.1f}h old')
    if stale:
        return 'STALE/INCOMPLETE: ' + '; '.join(stale)
    return f"OK: current={current_ds.strftime('%Y-%m-%d %H:%M')}; forecast_origin={origin.strftime('%Y-%m-%d %H:%M')}; oncall={pd.Timestamp(oncall_ds).strftime('%Y-%m-%d %H:%M')}"


def append_row(dbx, row, attempts=3):
    for attempt in range(attempts):
        metadata = None
        existing = []
        try:
            metadata, response = dbx.files_download(BLURB_PATH)
            text = response.content.decode('utf-8-sig')
            if text.strip():
                reader = csv.DictReader(io.StringIO(text))
                if reader.fieldnames != COLUMNS:
                    raise RuntimeError(f'Unexpected columns in {BLURB_PATH}: {reader.fieldnames}; refusing to overwrite')
                existing = [{c: r.get(c, '') for c in COLUMNS} for r in reader]
        except ApiError as err:
            if not is_not_found(err):
                raise

        if any(r['generated_at_local'] == row['generated_at_local'] for r in existing):
            print(f"Row {row['generated_at_local']} already exists; skipping duplicate")
            return False

        buf = io.StringIO(newline='')
        writer = csv.DictWriter(buf, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)
        mode = WriteMode.add if metadata is None else WriteMode.update(metadata.rev)
        try:
            result = dbx.files_upload(buf.getvalue().encode('utf-8'), BLURB_PATH, mode=mode, mute=True)
            print(f'Appended Dropbox row: {BLURB_PATH} rev={result.rev} rows={len(existing)+1}')
            return True
        except ApiError as err:
            if is_conflict(err) and attempt < attempts - 1:
                print('Concurrent Dropbox update detected; retrying append')
                continue
            raise
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enforce-schedule', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    now = datetime.now(TZ)
    if args.enforce_schedule and now.hour not in RUN_HOURS:
        print(f'Skipping inactive DST cron at {now.isoformat()}')
        return 0
    generated = now.replace(minute=45, second=0, microsecond=0) if args.enforce_schedule else now.replace(microsecond=0)
    now_naive = pd.Timestamp(now.replace(tzinfo=None))

    dbx = get_dbx()
    forecast, _ = download_csv(dbx, FORECAST_PATH)
    current, _ = download_csv(dbx, CURRENT_PATH)
    prob, _ = download_csv(dbx, ONCALL_PROB_PATH, required=False)
    impact, _ = download_csv(dbx, ONCALL_IMPACT_PATH, required=False)
    forecast, current = prep_forecast(forecast), prep_current(current)
    if forecast.empty or current.empty:
        raise RuntimeError('forecast-v2.1.csv and current.csv must both contain data')

    ds, metrics, room, prepod = latest_metrics(current)
    window = future_window(forecast, ds)
    recommendation, rationale, oncall_ds = oncall_assessment(prob, impact, now_naive)
    status = freshness(now_naive, ds, forecast, oncall_ds)

    pieces = []
    if status.startswith('STALE'):
        pieces.append('Data freshness warning: ' + status.replace('STALE/INCOMPLETE: ', '') + '.')
    anomaly = anomaly_text(forecast, ds, metrics)
    if anomaly:
        pieces.append(anomaly)
    if metrics['TTStr'] is not None:
        pieces.append(f"Stretcher occupancy is ~{metrics['TTStr'] / STRETCHER_CAPACITY * 100:.0f}% ({fmt(metrics['TTStr'])}/53).")
    overflow = overflow_text(metrics['Overflow'], room, prepod)
    if overflow:
        pieces.append(overflow)
    pieces.append(trajectory_text(window, metrics))
    staffing = staffing_text(window)
    if staffing:
        pieces.append(staffing)
    pieces.append(f'On-call: {recommendation} — {rationale}.')

    row = {
        'generated_at_local': generated.isoformat(),
        'forecast_data_time_local': ds.strftime('%Y-%m-%d %H:%M:%S'),
        'blurb': ' '.join(pieces),
        'oncall_recommendation': recommendation,
        'oncall_rationale': rationale,
        'source_status': status,
    }

    with open('ed_forecast_blurb_row.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    print(row['blurb'])
    print('source_status=' + status)
    if not args.dry_run:
        append_row(dbx, row)
    return 0


if __name__ == '__main__':
    sys.exit(main())
