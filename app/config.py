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

# Optional. Pollinations.ai (image generation) works fully anonymously with
# no key at all, but anonymous requests get their "pollinations.ai" logo
# stamped on every image and a lower rate limit. A free account at
# https://enter.pollinations.ai removes the watermark and raises the rate
# limit — paste that key here if you have one. Leaving this unset is fine;
# image generation still works, just with the watermark.
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")
