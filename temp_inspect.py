import sqlite3
import json

conn = sqlite3.connect('data/lumen.db')
cur = conn.cursor()
cur.execute('SELECT job_id, error FROM jobs WHERE error IS NOT NULL')
rows = cur.fetchall()
for row in rows:
    job_id, error = row
    if error and ('gemini' in error.lower() or 'api' in error.lower() or 'key' in error.lower()):
        print(f'Job {job_id} error contains relevant text:')
        print(error)
        print('---')
conn.close()