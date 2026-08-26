#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

from dropbox.exceptions import ApiError
from dropbox.files import WriteMode

from generate_ed_forecast_blurb import BLURB_PATH, get_dbx, is_conflict, is_not_found

LEGACY_COLUMNS = [
    'generated_at_local',
    'forecast_data_time_local',
    'blurb',
    'oncall_recommendation',
    'oncall_rationale',
    'source_status',
]

COLUMNS = [
    'generated_at_local',
    'forecast_data_time_local',
    'blurb_id',
    'blurb',
    'oncall_recommendation',
    'oncall_rationale',
    'send_recommended',
    'send_reason',
    'source_status',
]

REQUIRED_KEYS = set(COLUMNS)
ALLOWED_SEND_REASONS = {'ROUTINE', 'ONCALL_ALERT', 'ROUTINE_ONCALL', 'NONE'}


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {'true', '1', 'yes'}:
        return True
    if text in {'false', '0', 'no'}:
        return False
    raise RuntimeError('send_recommended must be true or false')


def load_request(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise RuntimeError('Request must be a JSON object')

    missing = REQUIRED_KEYS - set(payload)
    extra = set(payload) - REQUIRED_KEYS
    if missing or extra:
        raise RuntimeError(
            f'Request keys must exactly match CSV columns; missing={sorted(missing)} extra={sorted(extra)}'
        )

    send_recommended = parse_bool(payload['send_recommended'])
    send_reason = str(payload['send_reason']).strip().upper()
    if send_reason not in ALLOWED_SEND_REASONS:
        raise RuntimeError(f'send_reason must be one of {sorted(ALLOWED_SEND_REASONS)}')
    if send_recommended and send_reason == 'NONE':
        raise RuntimeError('send_reason cannot be NONE when send_recommended is true')
    if not send_recommended and send_reason != 'NONE':
        raise RuntimeError('send_reason must be NONE when send_recommended is false')

    row = {column: str(payload[column]).strip() for column in COLUMNS}
    row['send_recommended'] = 'true' if send_recommended else 'false'
    row['send_reason'] = send_reason

    for required in ['generated_at_local', 'forecast_data_time_local', 'blurb_id', 'blurb', 'source_status']:
        if not row[required]:
            raise RuntimeError(f'{required} is required')
    return row


def _read_existing(dbx):
    metadata = None
    existing: list[dict[str, str]] = []
    try:
        metadata, response = dbx.files_download(BLURB_PATH)
        text = response.content.decode('utf-8-sig')
        if text.strip():
            reader = csv.DictReader(io.StringIO(text))
            fieldnames = reader.fieldnames or []
            if fieldnames == COLUMNS:
                existing = [{c: r.get(c, '') for c in COLUMNS} for r in reader]
            elif fieldnames == LEGACY_COLUMNS:
                # One-time, lossless schema migration: preserve legacy rows and leave
                # delivery metadata blank because those historical sends are unknown.
                for legacy in reader:
                    migrated = {c: '' for c in COLUMNS}
                    for c in LEGACY_COLUMNS:
                        migrated[c] = legacy.get(c, '')
                    existing.append(migrated)
            else:
                raise RuntimeError(
                    f'Unexpected columns in {BLURB_PATH}: {fieldnames}; refusing to overwrite'
                )
    except ApiError as err:
        if not is_not_found(err):
            raise
    return metadata, existing


def append_row(dbx, row: dict[str, str], attempts: int = 3) -> bool:
    for attempt in range(attempts):
        metadata, existing = _read_existing(dbx)

        if any(r.get('blurb_id') and r.get('blurb_id') == row['blurb_id'] for r in existing):
            print(f"Blurb {row['blurb_id']} already exists; skipping duplicate")
            return False
        if any(r.get('generated_at_local') == row['generated_at_local'] for r in existing):
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Append an exact pre-generated ED forecast blurb request to Dropbox.'
    )
    parser.add_argument(
        '--request',
        default='.github/ed_forecast_blurb_request.json',
        help='Path to request JSON committed by the blurb author',
    )
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    row = load_request(Path(args.request))

    with open('ed_forecast_blurb_row.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    print(row['blurb'])
    print('blurb_id=' + row['blurb_id'])
    print('oncall_recommendation=' + row['oncall_recommendation'])
    print('send_recommended=' + row['send_recommended'])
    print('send_reason=' + row['send_reason'])
    print('source_status=' + row['source_status'])

    if args.dry_run:
        print('Dry run: request validated; Dropbox not modified.')
        return 0

    dbx = get_dbx()
    appended = append_row(dbx, row)
    print('append_result=' + ('appended' if appended else 'duplicate_skipped'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
