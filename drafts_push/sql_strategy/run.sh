for i in {1..7}
do
    for j in {1..5}
    do
        echo "$i - $j"
        DATABASE_URL="postgresql://formula_one:formula_one@127.0.0.1:5434/formula_one" python api_to_sql.py 2026 $i $j
    done
done