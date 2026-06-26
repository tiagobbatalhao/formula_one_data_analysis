clear:
	psql "postgresql://formula_one:formula_one@127.0.0.1:5434/formula_one" -f db_init.sql

run:
	DATABASE_URL="postgresql://formula_one:formula_one@127.0.0.1:5434/formula_one" python api_to_sql.py 2026 1 5