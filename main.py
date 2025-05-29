from dotenv import load_dotenv 
from app import create_app
import logging
import uvicorn

load_dotenv()

app = create_app()
logging.info("Sidemed AI - Server started")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
