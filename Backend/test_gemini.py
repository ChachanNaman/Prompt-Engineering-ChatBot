import os
import httpx
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
print("--- Starting Gemini API HTTP Test ---")

# Get the API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

async def main():
    if not GEMINI_API_KEY:
        print("🔴 Error: GEMINI_API_KEY not found in .env file.")
        return

    print("✅ API Key found.")
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": "Tell me a short, fun fact about computers."
            }]
        }]
    }
    
    headers = {"Content-Type": "application/json"}
    
    print("⚙️  Sending a direct HTTP request to Gemini...")
    print("   This test will time out after 20 seconds if it gets stuck.")

    try:
        # Use httpx with a timeout
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(api_url, json=payload, headers=headers)
            
            # Check for HTTP errors
            response.raise_for_status() 
            
            response_json = response.json()
            
            print("\n✅ Gemini Response:")
            print("--------------------")
            # Safely get the text from the response
            text = response_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No text found in response.')
            print(text)
            print("--------------------")

    except httpx.TimeoutException:
        print("\n🔴 An error occurred: Request timed out after 20 seconds.")
        print("   This strongly suggests a firewall, proxy, or network issue is blocking the connection to Google's servers.")
    except httpx.HTTPStatusError as e:
        print(f"\n🔴 An HTTP error occurred: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print("\n🔴 An unexpected error occurred:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())