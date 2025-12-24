import requests
import json

URL = "http://127.0.0.1:9001/zjh/ai/decide"

payload = {
    "ante_bb": 5.0,
    "round_index": 0,
    "current_bet_unit": 1,
    "first_round_open_unit": False,
    "players": [
        {
            "stack_bb": 120,
            "bet_bb": 5,
            "has_seen": False,
            "alive": True,
            "cards": [(14,0),(13,1),(12,2)]
        },
        {
            "stack_bb": 100,
            "bet_bb": 5,
            "has_seen": False,
            "alive": True,
            "cards": [(9,0),(9,1),(3,2)]
        },
        {
            "stack_bb": 80,
            "bet_bb": 5,
            "has_seen": False,
            "alive": True,
            "cards": [(2,0),(7,1),(11,2)]
        }
    ]
}

resp = requests.post(URL, json=payload)
print("Status:", resp.status_code)
print(json.dumps(resp.json(), indent=2))
