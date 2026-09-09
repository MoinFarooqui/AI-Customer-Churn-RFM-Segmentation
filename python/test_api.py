import os
from pathlib import Path
from dotenv import load_dotenv

# Get project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

# Check API key
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print("API key loaded successfully!")
else:
    print("API key was not found. Check your .env file.")