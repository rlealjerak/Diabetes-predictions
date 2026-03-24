init:
	python -c "import sqlite3; sqlite3.connect('db/diabetes_trends.db').executescript(open('db/schema.sql').read())"
