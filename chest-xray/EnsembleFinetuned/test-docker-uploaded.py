
import requests

# URL of the FastAPI endpoint
url = "http://medibot-ensemble-model-docker.icyflower-27afbae0.uaenorth.azurecontainerapps.io/predict"
# url = "http://220.174.46.7:8000/predict"

# Path to the image file
image_path = r"A:\Univeristy\Projects\Graduation-Project\Images Model\NIH Chest-Xrays\Perturbed\test\bcmci00027638_000.png"

# Open the image file in binary mode
with open(image_path, "rb") as image_file:
    # Prepare the payload with the image file
    files = {"file": image_file}
    
    # Send the POST request
    response = requests.post(url, files=files)
    
    # Print the raw response text
    print("Response Status Code:", response.status_code)
    print("Response Content:", response.text)
    
    try:
        # Print the response from the server
        print("Response JSON:", response.json())
    except requests.exceptions.JSONDecodeError:
        print("Failed to decode JSON from response")
