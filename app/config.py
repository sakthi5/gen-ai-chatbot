import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Read the Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Create a .env file in the project root "
        "with a line like: GROQ_API_KEY=your_key_here"
    )
