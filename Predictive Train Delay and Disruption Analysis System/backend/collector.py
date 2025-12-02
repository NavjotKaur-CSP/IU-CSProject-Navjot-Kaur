import requests
import csv
import time
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
FROM_STATION = "8011160"      # Berlin Hbf (example EVA ID)
TO_STATION = "0621103"        # Neukölln (example EVA ID)
RESULTS = 10                  # journeys per fetch
FETCH_INTERVAL = 300          # fetch every 5 minutes

CSV_FILE = "train_delays.csv"

API_URL = "https://v6.db.transport.rest/journeys"


# -----------------------------
# Ensure CSV file exists
# -----------------------------
def init_csv():
    try:
        with open(CSV_FILE, "x", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp_collected",
                "from_station",
                "to_station",
                "train_type",
                "line",
                "planned_departure",
                "actual_departure",
                "delay_minutes",
                "operator",
                "remarks",
                "raw_train_id"
            ])
        print("Created CSV file.")
    except FileExistsError:
        print("CSV file already exists. Appending data.")


# -----------------------------
# Fetch journeys from API
# -----------------------------
def fetch_journey_data():
    url = f"{API_URL}?from={FROM_STATION}&to={TO_STATION}&results={RESULTS}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return None

        return response.json()

    except Exception as e:
        print("Request failed:", e)
        return None


# -----------------------------
# Save records to CSV
# -----------------------------
def save_to_csv(journeys):
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        now = datetime.utcnow().isoformat()

        for j in journeys.get("journeys", []):
            leg = j["legs"][0]  # first main leg

            planned = leg["departure"]
            actual = leg.get("departureActual", planned)
            delay = leg.get("departureDelay", 0)

            train_type = leg.get("mode", "")
            line = leg.get("line", {}).get("name", "")
            operator = leg.get("operator", {}).get("name", "")
            remarks = "; ".join(r.get("text", "") for r in j.get("remarks", []))
            train_id = j.get("id", "")

            writer.writerow([
                now,
                FROM_STATION,
                TO_STATION,
                train_type,
                line,
                planned,
                actual,
                delay,
                operator,
                remarks,
                train_id,
            ])

        print(f"Saved {len(journeys.get('journeys', []))} records.")


# -----------------------------
# Main Loop
# -----------------------------
def start_collector():
    init_csv()

    while True:
        print("Fetching journeys...")
        data = fetch_journey_data()

        if data:
            save_to_csv(data)
        else:
            print("No data fetched.")

        print(f"Waiting {FETCH_INTERVAL} seconds...\n")
        time.sleep(FETCH_INTERVAL)


if __name__ == "__main__":
    start_collector()