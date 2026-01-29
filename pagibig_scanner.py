import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional

import requests
from bs4 import BeautifulSoup

from apscheduler.schedulers.background import BackgroundScheduler
import pytz

# ---------------- CONFIG ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

URLS = {
    1: "https://www.pagibigfundservices.com/Magpalistasa4ph/Project/Projects?loc=1",
    2: "https://www.pagibigfundservices.com/Magpalistasa4ph/Project/Projects?loc=2",
}

STATE_FILE = os.getenv("PAGIBIG_STATE_FILE", "pagibig_state.json")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

TZ_NAME = os.getenv("TZ", "Asia/Manila")
TZ = pytz.timezone(TZ_NAME)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

ASOF_REGEX = re.compile(r"As of\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", re.IGNORECASE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------- TELEGRAM ----------------
def tg_send(message: str) -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_TOKEN or CHAT_ID env vars.")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "disable_web_page_preview": True}
    r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()


# ---------------- STATE ----------------
def load_state() -> Dict:
    if not os.path.exists(STATE_FILE):
        return {"locs": {}}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"locs": {}}


def save_state(state: Dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ---------------- PARSING ----------------
def parse_asof_date(text: str) -> Optional[datetime]:
    m = ASOF_REGEX.search(text)
    if not m:
        return None
    s = m.group(1).strip()
    try:
        return datetime.strptime(s, "%B %d, %Y")
    except ValueError:
        return None


def extract_projects(soup: BeautifulSoup, base_url: str) -> Dict[str, Dict[str, str]]:
    projects: Dict[str, Dict[str, str]] = {}

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        name = " ".join(a.get_text(" ", strip=True).split())

        if not href or not name or len(name) < 3:
            continue
        if "javascript:" in href.lower():
            continue
        if "project" not in href.lower():
            continue

        if href.startswith("/"):
            full_url = base_url.rstrip("/") + href
        elif href.startswith("http"):
            full_url = href
        else:
            full_url = base_url.rstrip("/") + "/" + href

        id_match = re.search(r"(?i)(?:\b|[?&])(id|projectid|pid)=([^&]+)", full_url)
        key = f"{id_match.group(1).lower()}={id_match.group(2)}" if id_match else full_url

        if name.lower() in {"home", "projects", "back", "next", "previous"}:
            continue

        projects[key] = {"name": name, "url": full_url}

    return projects


def fetch_and_parse(url: str) -> Tuple[Optional[datetime], Dict[str, Dict[str, str]]]:
    r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    page_text = soup.get_text("\n", strip=True)

    asof_dt = parse_asof_date(page_text)
    projects = extract_projects(soup, base_url="https://www.pagibigfundservices.com")
    return asof_dt, projects


# ---------------- LOGIC ----------------
def format_new_projects(loc: int, asof: Optional[datetime], new_items: Dict[str, Dict[str, str]]) -> str:
    asof_str = asof.strftime("%B %d, %Y") if asof else "Unknown"
    lines = [f"🏠 Pag-IBIG 4PH Update (loc={loc})", f"📅 As of: {asof_str}", ""]

    if not new_items:
        lines.append("No new projects found (compared to saved list).")
        return "\n".join(lines)

    lines.append(f"✅ New projects found: {len(new_items)}\n")

    max_show = 25
    for i, item in enumerate(list(new_items.values())[:max_show], start=1):
        lines.append(f"{i}. {item['name']}")
        lines.append(f"   {item['url']}")

    if len(new_items) > max_show:
        lines.append(f"\n…plus {len(new_items) - max_show} more.")

    return "\n".join(lines)


def scan_job() -> None:
    state = load_state()

    for loc, url in URLS.items():
        try:
            logging.info("Scanning loc=%s ...", loc)
            asof_dt, projects = fetch_and_parse(url)

            loc_state = state.setdefault("locs", {}).setdefault(str(loc), {})
            prev_asof_str = loc_state.get("asof")
            prev_projects = loc_state.get("projects", {})

            new_keys = set(projects.keys()) - set(prev_projects.keys())
            new_items = {k: projects[k] for k in new_keys}

            should_notify = False

            if asof_dt and prev_asof_str:
                try:
                    prev_asof_dt = datetime.strptime(prev_asof_str, "%B %d, %Y")
                    if asof_dt > prev_asof_dt:
                        should_notify = True
                except ValueError:
                    pass

            if new_items:
                should_notify = True

            if should_notify:
                tg_send(format_new_projects(loc, asof_dt, new_items))
                logging.info("Notified loc=%s. New=%s", loc, len(new_items))
            else:
                logging.info("No changes worth notifying for loc=%s.", loc)

            # Save snapshot
            if asof_dt:
                loc_state["asof"] = asof_dt.strftime("%B %d, %Y")
            loc_state["projects"] = projects
            state["locs"][str(loc)] = loc_state
            save_state(state)

        except Exception as e:
            logging.exception("Error scanning loc=%s: %s", loc, e)
            try:
                tg_send(f"⚠️ Pag-IBIG scanner error (loc={loc}): {e}")
            except Exception:
                pass
def get_live_latest_summary() -> str:
    """
    LIVE fetch from Pag-IBIG site.
    Does NOT read/write pagibig_state.json.
    """
    lines = ["🟢 Pag-IBIG 4PH LIVE UPDATE", ""]

    for loc, url in URLS.items():
        try:
            asof_dt, projects = fetch_and_parse(url)
            asof_str = asof_dt.strftime("%B %d, %Y") if asof_dt else "Unknown"

            lines.append(f"🏠 Location: loc={loc}")
            lines.append(f"📅 As of: {asof_str}")
            lines.append(f"🧾 Projects found: {len(projects)}")
            lines.append("")

        except Exception as e:
            lines.append(f"🏠 Location: loc={loc}")
            lines.append(f"❌ Live fetch failed: {e}")
            lines.append("")

    return "\n".join(lines)


def main():
    # Send startup ping once
    try:
        tg_send("🤖 Pag-IBIG 4PH scanner started. Checking loc=1 and loc=2 every 4 hours.")
    except Exception as e:
        logging.warning("Startup Telegram ping failed: %s", e)

    # Background scheduler (non-blocking)
    sched = BackgroundScheduler(timezone=TZ)

    # Run immediately once, then every 4 hours
    sched.add_job(scan_job, "interval", hours=4, next_run_time=datetime.now(TZ), id="pagibig_scan")
    sched.start()

    # Keep module alive if run standalone
    logging.info("Pag-IBIG scheduler running (every 4 hours). TZ=%s", TZ_NAME)
    try:
        while True:
            import time
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
