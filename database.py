from pymongo import MongoClient
from config import MONGO_URI, DATABASE_NAME, logger

# Ensure required environment variables are set
if not MONGO_URI or not DATABASE_NAME:
    logger.error("Database configuration missing in environment variables.")
    raise ValueError("MONGO_URI and DATABASE_NAME must be set in the .env file.")

# Create a single MongoClient connection
try:
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000
    )

    client.admin.command("ping")

    db = client[DATABASE_NAME]

    users_collection = db["users"]
    patients_collection = db["patients"]
    reports_collection = db["reports"]
    chats_collection = db["chats"]
    historical_collection = db["historical"]

    logger.info(
        f"Connected to MongoDB database '{DATABASE_NAME}' successfully."
    )

except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise