"""
Notification system for TPU pipeline.

Sends alerts on: phase start/complete, abort, anomaly, daily heartbeat.

Configure via environment variables (set in tpu_master.sh):
  NOTIFY_EMAIL      — SMTP recipient (e.g. john.goodman@oceansparx.com)
  NOTIFY_SMTP_HOST  — SMTP host (default: smtp.gmail.com)
  NOTIFY_SMTP_PORT  — SMTP port (default: 587)
  NOTIFY_SMTP_USER  — SMTP login
  NOTIFY_SMTP_PASS  — SMTP password / app password
  NOTIFY_WEBHOOK    — Optional Slack/Discord/generic webhook URL
  GCS_BUCKET        — GCS bucket for result upload (e.g. gs://oceansparx-tpu)
  RUN_ID            — Run identifier (auto-set by master)

Usage:
  from notify import notify
  notify("PHASE_COMPLETE", "Phase 1 FlashOptim QM9 done", data={"mae": 0.051})
  notify("ABORT", "Checkpoint C failed — memory reduction only 12%", urgent=True)
  notify("HEARTBEAT", "All systems nominal", data=summary_dict)
"""

import os
import json
import time
import socket
import smtplib
import traceback
import urllib.request
import urllib.error
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

LOG_FILE = Path(os.environ.get("OUTPUT_DIR", "/tmp/flashoptim_results")) / "notify.log"
RUN_ID   = os.environ.get("RUN_ID", f"tpu_{datetime.now().strftime('%Y%m%d_%H%M')}")
HOSTNAME = socket.gethostname()


def _log(level: str, event: str, msg: str):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] [{event}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def _send_email(subject: str, body: str):
    host = os.environ.get("NOTIFY_SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("NOTIFY_SMTP_PORT", 587))
    user = os.environ.get("NOTIFY_SMTP_USER", "")
    pw   = os.environ.get("NOTIFY_SMTP_PASS", "")
    to   = os.environ.get("NOTIFY_EMAIL", "")

    if not (user and pw and to):
        return

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[AegisMind TPU | {RUN_ID}] {subject}"
        msg["From"]    = user
        msg["To"]      = to
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(host, port, timeout=30) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg)
        _log("INFO", "EMAIL", f"Sent to {to}: {subject}")
    except Exception as e:
        _log("WARN", "EMAIL", f"Failed: {e}")


def _send_webhook(event: str, msg: str, data: dict = None, urgent: bool = False):
    url = os.environ.get("NOTIFY_WEBHOOK", "")
    if not url:
        return

    payload = {
        "run_id":    RUN_ID,
        "host":      HOSTNAME,
        "event":     event,
        "message":   msg,
        "timestamp": datetime.now().isoformat(),
        "urgent":    urgent,
        "data":      data or {},
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            _log("INFO", "WEBHOOK", f"Posted {event} → {resp.status}")
    except Exception as e:
        _log("WARN", "WEBHOOK", f"Failed: {e}")


def notify(event: str, msg: str, data: dict = None, urgent: bool = False):
    """
    Send notification via all configured channels.

    Events (suggested conventions):
      PHASE_START     — entering a new phase
      PHASE_COMPLETE  — phase finished successfully
      CHECKPOINT      — checkpoint evaluation result
      ABORT           — experiment aborted (urgent=True auto-set)
      ANOMALY         — unexpected metric behaviour
      HEARTBEAT       — scheduled health check
      DONE            — entire pipeline finished
    """
    if event in ("ABORT",):
        urgent = True

    _log("INFO" if not urgent else "URGENT", event, msg)
    if data:
        _log("DATA", event, json.dumps(data, indent=2))

    subject = f"{'🚨 ' if urgent else ''}[{event}] {msg[:80]}"
    body = f"""
AegisMind TPU Pipeline Notification
=====================================
Run ID  : {RUN_ID}
Host    : {HOSTNAME}
Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Event   : {event}
Urgent  : {urgent}

Message:
{msg}

Data:
{json.dumps(data or {}, indent=2)}
"""

    smtp_user = os.environ.get("NOTIFY_SMTP_USER", "")
    smtp_pass = os.environ.get("NOTIFY_SMTP_PASS", "")
    smtp_to   = os.environ.get("NOTIFY_EMAIL", "")
    if smtp_user and smtp_pass and smtp_to:
        _send_email(subject, body)
    _send_webhook(event, msg, data=data, urgent=urgent)


def heartbeat(phase: str, epoch: int, metrics: dict):
    """Daily heartbeat — called from training loop."""
    notify("HEARTBEAT",
           f"Phase: {phase} | Epoch {epoch} | " +
           " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in list(metrics.items())[:5]),
           data={"phase": phase, "epoch": epoch, **metrics})
