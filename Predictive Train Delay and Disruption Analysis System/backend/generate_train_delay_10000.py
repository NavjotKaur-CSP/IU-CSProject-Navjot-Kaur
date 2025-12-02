# generate_train_delay_10000.py
# Generates a realistic synthetic DB-style train delay history CSV with 10,000 rows.
import csv, random
from datetime import datetime, timedelta

random.seed(42)

N = 10000
stations = [
    "Berlin Hbf","Alexanderplatz","Friedrichstraße","Zoologischer Garten","Lichtenberg","Ostbahnhof",
    "Berlin Südkreuz","Berlin Gesundbrunnen","Neukölln","Karlshorst","Pankow","Tempelhof",
    "Schöneberg","Spandau","Wannsee","Treptower Park","Prenzlauer Allee","Mitte","Charlottenburg",
    "Dahlem-Dorf","Adlershof","Märkisches Viertel","Hohenschönhausen","Grunewald","Wedding",
    "Ku'damm","Moabit","Heiligensee","Blankenburg","Hellersdorf"
]
products = ["ICE","IC","RE","RB","S-Bahn","U-Bahn"]
lines = ["S1","S2","S3","S5","S7","S9","U1","U2","U5","RE1","RE2","RB14","IC20","ICE10","RB21"]
delay_causes = [
    "Signal failure","Track maintenance","Weather","Staff shortage","Late incoming vehicle",
    "Technical fault","Congestion","Passenger incident","Operational change","Unknown"
]
notes_examples = [
    "Replacement bus", "No further information", "Affected by earlier disruption",
    "Cleared", "Policing support required", "Short delay at previous station",
    "Recovered en route", ""
]

start_date = datetime(2020,1,1)
end_date = datetime(2023,12,31)
days_range = (end_date - start_date).days

out_path = "train_delay_history_10000.csv"
with open(out_path, "w", newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow([
        "date","time","train_id","line","product","station","direction",
        "planned_delay_min","actual_delay_min","delay_cause","note"
    ])
    for i in range(N):
        d = start_date + timedelta(days=random.randint(0, days_range))
        dt = d + timedelta(seconds=random.randint(0, 24*3600-1))
        prod = random.choice(products)
        line = random.choice(lines)
        station = random.choice(stations)
        if prod in ("ICE","IC","RE","RB"):
            tid = f"{prod} {random.randint(100,999)}"
        else:
            tid = f"{prod} {random.choice(['A','B','C','D'])}{random.randint(1,99)}"
        p_delay = int(random.choices([0,1,2,3,5,10],[80,8,5,3,2,1])[0])
        dev = int(random.gauss(5 if p_delay>0 else 0, 8))
        a_delay = p_delay + dev
        if random.random() < 0.01:
            a_delay += random.randint(30,180)
        a_delay = max(-5, min(300, a_delay))
        if a_delay <= 0:
            cause = "On time / early"
        elif a_delay <= 3:
            cause = "Minor operational"
        else:
            # weighted-ish pick
            cause = random.choice(delay_causes)
        note = random.choice(notes_examples) if random.random() < 0.4 else ""
        w.writerow([
            dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"), tid, line, prod,
            station, random.choice(["inbound","outbound","towards Hbf","from Hbf"]),
            p_delay, a_delay, cause, note
        ])

print("Saved:", out_path, "Rows:", N) 