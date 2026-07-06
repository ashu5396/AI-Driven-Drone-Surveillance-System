import sqlite3
import threading


DB_NAME="alerts.db"
DB_LOCK=threading.Lock()

def init_db():
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    c.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        camera_id INTEGER,
        drone_id INTEGER,
        event_type TEXT,
        video_path TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_event(timestamp, cam_id, drone_id, event_type, video_path):
    with DB_LOCK:

        conn = sqlite3.connect(DB_NAME, timeout=10)
        c = conn.cursor()
        c.execute("PRAGMA journal_mode=WAL;")

        c.execute("""
        INSERT INTO events (timestamp, camera_id, drone_id, event_type, video_path)
        VALUES (?, ?, ?, ?, ?)
        """, (timestamp, cam_id, drone_id, event_type, video_path))

        conn.commit()
        conn.close()


def get_events():
    
        
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")

    c.execute("SELECT * FROM events ORDER BY id DESC LIMIT 25")
    rows = c.fetchall()

    conn.close()
    return rows
