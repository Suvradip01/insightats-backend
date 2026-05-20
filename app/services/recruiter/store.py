from __future__ import annotations

import time
from typing import Optional, Tuple
import sqlite3

from app.db import get_db_connection, ensure_db
from app.services.recruiter.security import new_token

_connect = get_db_connection


def create_recruiter(company: str, username: str, password_hash: str) -> int:
    ensure_db()
    now = int(time.time())
    con = _connect()
    try:
        cur = con.execute(
            "INSERT INTO recruiters(company, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (company.strip(), username.strip(), password_hash, now),
        )
        con.commit()
        return int(cur.lastrowid)
    finally:
        con.close()


def get_recruiter_by_company_username(company: str, username: str) -> Optional[sqlite3.Row]:
    ensure_db()
    con = _connect()
    try:
        cur = con.execute(
            "SELECT id, company, username, password_hash FROM recruiters WHERE company=? AND username=?",
            (company.strip(), username.strip()),
        )
        return cur.fetchone()
    finally:
        con.close()


def get_recruiter_by_username(username: str) -> Optional[sqlite3.Row]:
    ensure_db()
    con = _connect()
    try:
        cur = con.execute(
            "SELECT id, company, username, password_hash FROM recruiters WHERE username=?",
            (username.strip(),),
        )
        return cur.fetchone()
    finally:
        con.close()


def create_session(recruiter_id: int, *, ttl_seconds: int = 60 * 60 * 12) -> str:
    ensure_db()
    token = new_token()
    now = int(time.time())
    exp = now + int(ttl_seconds)
    con = _connect()
    try:
        con.execute(
            "INSERT INTO recruiter_sessions(token, recruiter_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (token, recruiter_id, exp, now),
        )
        con.commit()
    finally:
        con.close()
    return token


def get_principal_for_token(token: str) -> Optional[Tuple[int, str, str]]:
    """
    Returns (recruiter_id, company, username) if the session is valid.
    """
    ensure_db()
    now = int(time.time())
    con = _connect()
    try:
        cur = con.execute(
            """
            SELECT r.id as recruiter_id, r.company as company, r.username as username, s.expires_at as expires_at
            FROM recruiter_sessions s
            JOIN recruiters r ON r.id = s.recruiter_id
            WHERE s.token = ?
            """,
            (token,),
        )
        row = cur.fetchone()
        if not row:
            return None
        if int(row["expires_at"]) < now:
            con.execute("DELETE FROM recruiter_sessions WHERE token=?", (token,))
            con.commit()
            return None
        return int(row["recruiter_id"]), str(row["company"]), str(row["username"])
    finally:
        con.close()
