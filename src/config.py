import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Fetch the Twitter Bearer Token from the environment
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
