

import sqlite3
from typing import Optional, List, Dict, Any
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'queries.db')

def init_db():
	conn = sqlite3.connect(DB_PATH)
	c = conn.cursor()
	c.execute('''
		CREATE TABLE IF NOT EXISTS queries (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			question TEXT NOT NULL,
			answer TEXT,
			timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
			country TEXT,
			topic TEXT,
			user_metadata TEXT
		)
	''')
	conn.commit()
	conn.close()

def save_query(question: str, answer: str, country: Optional[str]=None, topic: Optional[str]=None, user_metadata: Optional[str]=None):
    answer_str = str(answer)  # Ensure answer is a string
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO queries (question, answer, country, topic, user_metadata)
        VALUES (?, ?, ?, ?, ?)
    ''', (question, answer_str, country, topic, user_metadata))
    conn.commit()
    conn.close()
def get_queries(country: Optional[str]=None, topic: Optional[str]=None, limit: int=50) -> List[Dict[str, Any]]:
	conn = sqlite3.connect(DB_PATH)
	c = conn.cursor()
	query = 'SELECT id, question, answer, timestamp, country, topic, user_metadata FROM queries WHERE 1=1'
	params = []
	if country:
		query += ' AND country = ?'
		params.append(country)
	if topic:
		query += ' AND topic = ?'
		params.append(topic)
	query += ' ORDER BY timestamp DESC LIMIT ?'
	params.append(limit)
	c.execute(query, params)
	rows = c.fetchall()
	conn.close()
	return [
		{
			'id': row[0],
			'question': row[1],
			'answer': row[2],
			'timestamp': row[3],
			'country': row[4],
			'topic': row[5],
			'user_metadata': row[6],
		}
		for row in rows
	]

def get_query_count() -> int:
	conn = sqlite3.connect(DB_PATH)
	c = conn.cursor()
	c.execute('SELECT COUNT(*) FROM queries')
	count = c.fetchone()[0]
	conn.close()
	return count

