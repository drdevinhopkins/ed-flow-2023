#!/usr/bin/env python3

"""
Watch Dropbox for changes to hourlyreport.pdf and trigger the local ED flow workflow.

Assumptions:
- Repo path: /home/dhopkins/apps/ed-flow-2023
- Existing .env contains:
    DROPBOX_APP_KEY
    DROPBOX_APP_SECRET
    DROPBOX_REFRESH_TOKEN
- Dropbox app is app-folder scoped, so:
    Dropbox/Apps/ed-flow-2023/hourlyreport.pdf
  appears to the Dropbox API as:
    /hourlyreport.pdf
- When the PDF changes, this runs:
    scripts/run_ed_flow_update.sh
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import dropbox
from dropbox.files import DeletedMetadata, FileMetadata


REPO_DIR = Path("/home/dhopkins/apps/ed-flow-2023")
ENV_FILE = REPO_DIR / ".env"
STATE_DIR = REPO_DIR / "state"
CURSOR_FILE = STATE_DIR / "dropbox_cursor.txt"

# App-folder Dropbox root. For Dropbox API, the app folder root is "".
WATCH_FOLDER = ""

# In Dropbox UI/local sync this is:
# Dropbox/Apps/ed-flow-2023/hourlyreport.pdf
# But through an app-folder Dropbox API token, it is:
TARGET_PDF = "/hourlyreport.pdf"

RUN_SCRIPT = REPO_DIR / "scripts" / "run_ed_flow_update.sh"

LONGPOLL_TIMEOUT_SECONDS = 120
ERROR_SLEEP_SECONDS = 30


def load_env_file(path: Path) -> None:
    """
    Minimal .env loader so this script works even if python-dotenv is not installed.

    Supports lines like:
        KEY=value
        KEY="value"
        KEY='value'
    Ignores blank lines and comments.
    Does not override already-exported environment variables.
    """
    if not path.exists():
        print(f"Warning: .env file not found at {path}", flush=True)
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        # Remove simple surrounding quotes.
        if len(value) >= 2:
            if (value[0] == value[-1]) and value[0] in ("'", '"'):
                value = value[1:-1]

        os.environ.setdefault(key, value)


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def make_dropbox_client() -> dropbox.Dropbox:
    app_key = required_env("DROPBOX_APP_KEY")
    app_secret = required_env("DROPBOX_APP_SECRET")
    refresh_token = required_env("DROPBOX_REFRESH_TOKEN")

    return dropbox.Dropbox(
        oauth2_refresh_token=refresh_token,
        app_key=app_key,
        app_secret=app_secret,
    )


def get_initial_cursor(dbx: dropbox.Dropbox) -> str:
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if CURSOR_FILE.exists():
        cursor = CURSOR_FILE.read_text().strip()
        if cursor:
            print(f"Using saved Dropbox cursor from {CURSOR_FILE}", flush=True)
            return cursor

    print("No saved cursor found. Creating latest Dropbox cursor.", flush=True)

    result = dbx.files_list_folder_get_latest_cursor(
        WATCH_FOLDER,
        recursive=False,
    )

    CURSOR_FILE.write_text(result.cursor)
    return result.cursor


def save_cursor(cursor: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CURSOR_FILE.write_text(cursor)


def run_workflow() -> None:
    if not RUN_SCRIPT.exists():
        raise RuntimeError(f"Workflow script not found: {RUN_SCRIPT}")

    print(f"Running workflow: {RUN_SCRIPT}", flush=True)

    subprocess.run(
        [str(RUN_SCRIPT)],
        cwd=str(REPO_DIR),
        check=True,
    )

    print("Workflow completed successfully.", flush=True)


def process_dropbox_changes(dbx: dropbox.Dropbox, cursor: str) -> str:
    """
    Pull all changes since cursor.
    If /hourlyreport.pdf changed, run the local workflow.
    Return the updated cursor.
    """
    result = dbx.files_list_folder_continue(cursor)

    target_changed = False
    target_deleted = False

    while True:
        for entry in result.entries:
            entry_path = getattr(entry, "path_lower", "")

            if entry_path == TARGET_PDF.lower():
                if isinstance(entry, FileMetadata):
                    target_changed = True
                    print(
                        f"Detected updated target PDF: {entry.path_display}; "
                        f"rev={entry.rev}; size={entry.size}",
                        flush=True,
                    )

                elif isinstance(entry, DeletedMetadata):
                    target_deleted = True
                    print(
                        f"Detected deletion of target PDF: {entry.path_display}",
                        flush=True,
                    )

        cursor = result.cursor

        if not result.has_more:
            break

        result = dbx.files_list_folder_continue(cursor)

    save_cursor(cursor)

    if target_deleted:
        raise RuntimeError(f"Target PDF was deleted from Dropbox: {TARGET_PDF}")

    if target_changed:
        run_workflow()
    else:
        print("Dropbox changed, but hourlyreport.pdf did not change.", flush=True)

    return cursor


def sanity_check_target_exists(dbx: dropbox.Dropbox) -> None:
    """
    Optional startup check. This confirms that the Dropbox API sees /hourlyreport.pdf.
    """
    try:
        metadata = dbx.files_get_metadata(TARGET_PDF)
        print(
            f"Watching Dropbox file: {metadata.path_display}; "
            f"rev={getattr(metadata, 'rev', 'unknown')}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"Warning: could not confirm target file exists at {TARGET_PDF}: {exc}",
            flush=True,
        )
        print(
            "If your Dropbox app is app-folder scoped, the path should usually be "
            "/hourlyreport.pdf, not /Apps/ed-flow-2023/hourlyreport.pdf.",
            flush=True,
        )


def main() -> int:
    load_env_file(ENV_FILE)

    print("Starting ED Flow Dropbox watcher.", flush=True)
    print(f"Repo: {REPO_DIR}", flush=True)
    print(f"Dropbox watch folder: app root", flush=True)
    print(f"Dropbox target PDF: {TARGET_PDF}", flush=True)

    dbx = make_dropbox_client()
    sanity_check_target_exists(dbx)

    cursor = get_initial_cursor(dbx)

    while True:
        try:
            poll_result = dbx.files_list_folder_longpoll(
                cursor,
                timeout=LONGPOLL_TIMEOUT_SECONDS,
            )

            if poll_result.changes:
                print("Dropbox reported changes. Checking changed files.", flush=True)
                cursor = process_dropbox_changes(dbx, cursor)
            else:
                print("No Dropbox changes during longpoll window.", flush=True)

            if poll_result.backoff:
                print(
                    f"Dropbox requested backoff: sleeping {poll_result.backoff} seconds.",
                    flush=True,
                )
                time.sleep(poll_result.backoff)

        except KeyboardInterrupt:
            print("Watcher stopped by keyboard interrupt.", flush=True)
            return 0

        except Exception as exc:
            print(f"Watcher error: {exc}", flush=True)
            time.sleep(ERROR_SLEEP_SECONDS)


if __name__ == "__main__":
    sys.exit(main())