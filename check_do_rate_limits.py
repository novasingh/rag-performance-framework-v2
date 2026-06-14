import requests
import os

# Grab the API key from config if available, otherwise from env
try:
    from rq1_experiment.config import DO_API_KEY
except ImportError:
    DO_API_KEY = os.environ.get("DO_API_KEY")

def check_rate_limits():
    print("=== DigitalOcean Inference Rate Limit Diagnostics ===")
    if not DO_API_KEY:
        print("Error: No DO_API_KEY found.")
        return

    url = "https://inference.do-ai.run/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DO_API_KEY}"
    }
    data = {
        "model": "llama-4-maverick",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        
        print(f"Status Code: {response.status_code}")
        print("\n--- All Response Headers ---")
        for key, value in response.headers.items():
            print(f"{key}: {value}")
            
        print("\n--- Rate Limit Specific Headers ---")
        rate_limit_keys = [k for k in response.headers.keys() if 'ratelimit' in k.lower() or 'quota' in k.lower() or 'limit' in k.lower() or 'remain' in k.lower() or 'reset' in k.lower()]
        
        if not rate_limit_keys:
            print("No standard rate limit headers (e.g., X-RateLimit-Remaining) were returned by DigitalOcean.")
        else:
            for k in rate_limit_keys:
                print(f"{k}: {response.headers[k]}")
                
        if response.status_code == 429:
            print("\nWARNING: Currently Rate Limited (HTTP 429)!")
            if 'Retry-After' in response.headers:
                print(f"Must wait: {response.headers['Retry-After']} seconds.")
                
    except Exception as e:
        print(f"Error checking API: {e}")

if __name__ == "__main__":
    check_rate_limits()
