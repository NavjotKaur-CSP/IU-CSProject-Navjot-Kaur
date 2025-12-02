import os
import requests
from datetime import datetime

CLIENT_ID = os.getenv("DB_CLIENT_ID")
CLIENT_SECRET = os.getenv("DB_CLIENT_SECRET")

eva = os.getenv("TEST_EVA") or "8011160"

now = datetime.now()
YYYY = now.strftime("%Y")
MM   = now.strftime("%m")
DD   = now.strftime("%d")
HH   = now.strftime("%H")

candidates = [
    f"https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1.0.213/plan/{eva}/{YYYY}/{MM}/{DD}/{HH}",
    f"https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1.0.213/plan/{eva}/{YYYY}/{MM}/{DD}",
    f"https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1.0.213/plan/{eva}",
    f"https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1/plan/{eva}/{YYYY}/{MM}/{DD}/{HH}",
    f"https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1/r/{eva}",
]

header_variants = [
    {"DB-Client-Id": CLIENT_ID, "DB-Api-Key": CLIENT_SECRET},
    {"DB-Client-Id": CLIENT_ID, "DB-Api-Key": CLIENT_SECRET, "Accept": "application/xml"},
    {"DB-Client-Id": CLIENT_ID, "DB-Api-Key": CLIENT_SECRET, "Accept": "application/json"},
]

def main():
    print("CLIENT_ID:", CLIENT_ID)
    print("CLIENT_SECRET:", CLIENT_SECRET[:6] + "***********")
    print("Testing EVA:", eva)
    print()

    for url in candidates:
        print("\nTrying URL:", url)
        for headers in header_variants:
            print("  Headers:", headers)
            try:
                r = requests.get(url, headers=headers)
                print("    Status:", r.status_code)
                print("    Body:", r.text[:400])
            except Exception as e:
                print("    ERROR:", e)

if __name__ == "__main__":
    main()

