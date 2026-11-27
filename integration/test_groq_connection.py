
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Find the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check your root .env file."
    )

# Initialize Groq client
client = Groq(api_key=api_key)

# Send a test request
try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: GROQ_CONNECTION_OK"
            }
        ],
        max_completion_tokens=100
    )

    print("\n===== GROQ CONNECTION TEST =====")
    print("Model:", "openai/gpt-oss-120b")
    print("Response:", response.choices[0].message.content)
    print("Groq API connection successful!")

except Exception as error:
    print("\nGroq connection failed.")
    print(type(error).__name__, error)
