#!/usr/bin/env python3
"""Fetch the official JGH RAMQ establishment holiday calendar.

RAMQ lets each establishment move the standard 13 professional holidays.  This utility
uses the official RAMQ calendar page and selects Montréal (06) and establishment 0011X
(Hôpital général juif).  It intentionally discovers select controls by their option text
instead of relying on fragile ASP.NET-generated element ids.

The script can run in diagnostic mode (default) and writes the final page text plus a
JSON description of the form controls.  Once the page structure is confirmed, the same
utility is used to parse and persist the exact JGH dates consumed by the forecasting
feature builder.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from playwright.async_api import async_playwright

RAMQ_URL = (
    "https://www4.prod.ramq.gouv.qc.ca/ETA/SN/SNC_CalenFerie/"
    "SNC2_CnsulCalenFerieEtab_iut/CnsulCalenFerieEtab.aspx"
)
REGION_CODE = "06"
ESTABLISHMENT_CODE = "0011X"


def _norm(text: str) -> str:
    return " ".join((text or "").replace("\xa0", " ").split())


async def _select_by_option_text(page, needle: str) -> dict[str, str]:
    """Select the first <select> containing an option whose text contains needle."""
    selects = page.locator("select")
    for i in range(await selects.count()):
        select = selects.nth(i)
        options = select.locator("option")
        for j in range(await options.count()):
            option = options.nth(j)
            text = _norm(await option.inner_text())
            value = await option.get_attribute("value") or ""
            if needle.casefold() in text.casefold():
                await select.select_option(value=value)
                await page.wait_for_load_state("networkidle")
                return {
                    "select_index": str(i),
                    "select_name": await select.get_attribute("name") or "",
                    "select_id": await select.get_attribute("id") or "",
                    "option_text": text,
                    "option_value": value,
                }
    raise RuntimeError(f"Could not find option containing {needle!r}")


async def _describe_selects(page) -> list[dict[str, object]]:
    described: list[dict[str, object]] = []
    selects = page.locator("select")
    for i in range(await selects.count()):
        select = selects.nth(i)
        options = []
        option_nodes = select.locator("option")
        for j in range(await option_nodes.count()):
            option = option_nodes.nth(j)
            options.append(
                {
                    "text": _norm(await option.inner_text()),
                    "value": await option.get_attribute("value") or "",
                    "selected": await option.is_checked() if await option.get_attribute("selected") else False,
                }
            )
        described.append(
            {
                "index": i,
                "name": await select.get_attribute("name") or "",
                "id": await select.get_attribute("id") or "",
                "options": options,
            }
        )
    return described


async def fetch(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(locale="fr-CA")
        await page.goto(RAMQ_URL, wait_until="networkidle", timeout=120_000)

        initial_selects = await _describe_selects(page)
        region_selection = await _select_by_option_text(page, REGION_CODE)
        after_region_selects = await _describe_selects(page)
        establishment_selection = await _select_by_option_text(page, ESTABLISHMENT_CODE)
        after_establishment_selects = await _describe_selects(page)

        body_text = _norm(await page.locator("body").inner_text())
        html = await page.content()

        # Keep a broad date scan in diagnostics.  Parsing to the canonical CSV is added
        # once the official page's returned format has been confirmed in CI.
        date_patterns = {
            "dmy_slash": sorted(set(re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", body_text))),
            "ymd_dash": sorted(set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", body_text))),
            "dmy_dash": sorted(set(re.findall(r"\b\d{1,2}-\d{1,2}-\d{4}\b", body_text))),
        }

        diagnostic = {
            "url": page.url,
            "region_selection": region_selection,
            "establishment_selection": establishment_selection,
            "initial_selects": initial_selects,
            "after_region_selects": after_region_selects,
            "after_establishment_selects": after_establishment_selects,
            "date_candidates": date_patterns,
        }
        (output_dir / "ramq-jgh-diagnostic.json").write_text(
            json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "ramq-jgh-page.txt").write_text(body_text + "\n", encoding="utf-8")
        (output_dir / "ramq-jgh-page.html").write_text(html, encoding="utf-8")
        await page.screenshot(path=str(output_dir / "ramq-jgh-page.png"), full_page=True)
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output-ramq"))
    args = parser.parse_args()
    asyncio.run(fetch(args.output_dir))
