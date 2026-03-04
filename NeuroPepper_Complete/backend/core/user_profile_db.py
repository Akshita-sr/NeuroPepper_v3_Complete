# backend/core/user_profile_db.py
# ============================================
# User Profile Database
# ============================================
# A simple SQLite database that stores user profiles.
# These profiles provide context to LLMs during:
# - Conversations (the model knows who it's talking to)
# - Mental-state detection (age group, preferences)
# - Evaluation sessions (track which user rated what)
# ============================================

import os
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
load_dotenv()

DB_PATH = Path(os.getenv("USER_PROFILE_DB", "data/user_profiles/profiles.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class UserProfile:
    """One user's profile."""
    user_id: str
    name: str
    age_group: str = "adult"  # "child", "teen", "adult", "senior"
    language: str = "en"
    preferences: str = "{}"   # JSON string of preferences
    context_notes: str = ""   # Extra notes for the LLM
    created_at: str = ""
    updated_at: str = ""

    def preferences_dict(self) -> Dict:
        try:
            return json.loads(self.preferences)
        except Exception:
            return {}

    def to_system_context(self) -> str:
        """
        Generate a system prompt snippet that tells the LLM
        about this user. Used to personalize responses.
        """
        prefs = self.preferences_dict()
        parts = [f"You are talking to {self.name}."]
        if self.age_group:
            parts.append(f"Age group: {self.age_group}.")
        if self.language != "en":
            parts.append(f"Preferred language: {self.language}.")
        if prefs:
            parts.append(f"User preferences: {json.dumps(prefs)}.")
        if self.context_notes:
            parts.append(f"Additional context: {self.context_notes}")
        return " ".join(parts)


# ============================================
# DATABASE SETUP
# ============================================

def _get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with row_factory enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the tables if they don't exist."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age_group TEXT DEFAULT 'adult',
            language TEXT DEFAULT 'en',
            preferences TEXT DEFAULT '{}',
            context_notes TEXT DEFAULT '',
            created_at TEXT,
            updated_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            model_id TEXT,
            task_type TEXT,
            prompt TEXT,
            response TEXT,
            user_rating INTEGER,
            user_feedback TEXT,
            auto_scores TEXT,
            latency REAL,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
        )
    """)

    conn.commit()
    conn.close()


# Auto-initialize on import
init_db()


# ============================================
# CRUD OPERATIONS
# ============================================

def create_user(user_id: str, name: str, age_group: str = "adult",
                language: str = "en", preferences: Dict = None,
                context_notes: str = "") -> UserProfile:
    """Create a new user profile."""
    now = datetime.now().isoformat()
    prefs_json = json.dumps(preferences or {})

    conn = _get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO user_profiles
        (user_id, name, age_group, language, preferences, context_notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, name, age_group, language, prefs_json, context_notes, now, now))
    conn.commit()
    conn.close()

    return UserProfile(
        user_id=user_id, name=name, age_group=age_group,
        language=language, preferences=prefs_json,
        context_notes=context_notes, created_at=now, updated_at=now,
    )


def get_user(user_id: str) -> Optional[UserProfile]:
    """Get a user by ID."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()

    if row is None:
        return None
    return UserProfile(**dict(row))


def list_users() -> List[UserProfile]:
    """List all user profiles."""
    conn = _get_connection()
    rows = conn.execute("SELECT * FROM user_profiles ORDER BY name").fetchall()
    conn.close()
    return [UserProfile(**dict(r)) for r in rows]


def update_user(user_id: str, **kwargs) -> Optional[UserProfile]:
    """Update fields on a user profile."""
    user = get_user(user_id)
    if user is None:
        return None

    allowed_fields = {"name", "age_group", "language", "preferences", "context_notes"}
    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

    if "preferences" in updates and isinstance(updates["preferences"], dict):
        updates["preferences"] = json.dumps(updates["preferences"])

    updates["updated_at"] = datetime.now().isoformat()

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [user_id]

    conn = _get_connection()
    conn.execute(f"UPDATE user_profiles SET {set_clause} WHERE user_id = ?", values)
    conn.commit()
    conn.close()

    return get_user(user_id)


def delete_user(user_id: str) -> bool:
    """Delete a user profile."""
    conn = _get_connection()
    cursor = conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


# ============================================
# EVALUATION SESSION TRACKING
# ============================================

def save_eval_session(session_id: str, user_id: str, model_id: str,
                      task_type: str, prompt: str, response: str,
                      user_rating: int = 0, user_feedback: str = "",
                      auto_scores: Dict = None, latency: float = 0.0):
    """Save one evaluation session (one prompt+response from one model)."""
    conn = _get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO evaluation_sessions
        (session_id, user_id, model_id, task_type, prompt, response,
         user_rating, user_feedback, auto_scores, latency, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id, user_id, model_id, task_type, prompt, response,
        user_rating, user_feedback, json.dumps(auto_scores or {}),
        latency, datetime.now().isoformat(),
    ))
    conn.commit()
    conn.close()


def get_eval_sessions(user_id: str = None, model_id: str = None,
                      task_type: str = None) -> List[Dict]:
    """Query evaluation sessions with optional filters."""
    conn = _get_connection()
    query = "SELECT * FROM evaluation_sessions WHERE 1=1"
    params = []

    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)
    if model_id:
        query += " AND model_id = ?"
        params.append(model_id)
    if task_type:
        query += " AND task_type = ?"
        params.append(task_type)

    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_eval_summary() -> Dict:
    """Get a summary of all evaluations grouped by model and task."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT model_id, task_type,
               COUNT(*) as total,
               AVG(user_rating) as avg_rating,
               AVG(latency) as avg_latency
        FROM evaluation_sessions
        GROUP BY model_id, task_type
        ORDER BY model_id, task_type
    """).fetchall()
    conn.close()

    summary = {}
    for r in rows:
        r = dict(r)
        key = r["model_id"]
        if key not in summary:
            summary[key] = {}
        summary[key][r["task_type"]] = {
            "total": r["total"],
            "avg_rating": round(r["avg_rating"] or 0, 2),
            "avg_latency": round(r["avg_latency"] or 0, 3),
        }
    return summary


# ============================================
# SEED DATA (for testing)
# ============================================

def seed_sample_users():
    """Create a few sample user profiles for testing."""
    samples = [
        ("user_adult_01", "Alice", "adult", "en",
         {"tone": "formal", "topics": ["technology", "science"]},
         "Works as a software engineer. Prefers detailed explanations."),
        ("user_adult_02", "Bob", "adult", "en",
         {"tone": "casual", "topics": ["sports", "cooking"]},
         "Casual user. Prefers short, simple answers."),
        ("user_teen_01", "Charlie", "teen", "en",
         {"tone": "friendly", "topics": ["gaming", "school"]},
         "High school student. Uses informal language."),
        ("user_child_01", "Daisy", "child", "en",
         {"tone": "simple", "topics": ["animals", "cartoons"]},
         "Young child. Needs very simple vocabulary."),
        ("user_senior_01", "Edward", "senior", "en",
         {"tone": "patient", "topics": ["health", "gardening"]},
         "Retired teacher. Appreciates patience and clarity."),
    ]
    for uid, name, age, lang, prefs, notes in samples:
        create_user(uid, name, age, lang, prefs, notes)
    print(f"✅ Seeded {len(samples)} sample user profiles")
