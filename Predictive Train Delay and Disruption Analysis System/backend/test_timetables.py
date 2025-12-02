import requests
import os
from datetime import datetime

CLIENT_ID = os.getenv("ab13d17f9136410fa179566a31381eb7")
CLIENT_SECRET = os.getenv("7eca39785e381ea9d52bbf4a6832396a")

# Berlin Hbf
eva = "8011160"

# Current date/time
now = datetime.now()
YYYY = now.strftime("%Y")
MM   = now.strftime("%m")
DD   = now.strftime("%d")
HH   = now.strftime("%H")

url = f"https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1.0.213/plan/{eva}/{YYYY}/{MM}/{DD}/{HH}"

headers = {
    "DB-Client-Id": CLIENT_ID,
    "DB-Api-Key": CLIENT_SECRET,
    "Accept": "application/xml"
}

print("Requesting:", url)

response = requests.get(url, headers=headers)

print("Status:", response.status_code)
print(response.text)