import requests

url = "http://127.0.0.1:5000/api/start"
payload = {"student_id": "test", "screen_time": ""}

try:
    res = requests.post(url, json=payload)
    print("Status:", res.status_code)
    print("Response:", res.text)
except Exception as e:
    print("Exception:", e)
