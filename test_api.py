import requests
import sys

API = "http://127.0.0.1:8000"

def test_folder():
    # 1. Get latest file_id
    try:
        # Use an existing ID from previous logs (e.g. 43)
        file_id = 43
        print(f"Testing with file_id: {file_id}")
        
        # 2. Create folder
        payload = {"category": "TestCat", "subject": "TestSub"}
        r = requests.post(f"{API}/messages/{file_id}/folders", json=payload)
        print(f"Create status: {r.status_code}, Response: {r.text}")
        
        # 3. Delete folder
        r = requests.delete(f"{API}/messages/{file_id}/folders", params={"category": "TestCat", "subject": "TestSub"})
        print(f"Delete status: {r.status_code}, Response: {r.text}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_folder()
