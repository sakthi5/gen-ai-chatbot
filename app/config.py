import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Read the Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")