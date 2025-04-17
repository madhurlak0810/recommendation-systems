import requests

# Define the base URL for your FastAPI app
url = "http://127.0.0.1:8000/popular-products"

# Define the parameters (query string)
params = {
    "top_n": 5,  # Get the top 5 products
    "main_category": "music",  # Filter by main category
    "sub_category": "Musical Instruments & Professional Audio"  # Filter by sub category (optional)
}

# Send the GET request to the API
response = requests.get(url, params=params)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    print("Popular Products:", data)
else:
    print(f"Error: {response.status_code} - {response.text}")