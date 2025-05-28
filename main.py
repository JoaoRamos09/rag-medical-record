from dotenv import load_dotenv 
from app import create_app
import logging

load_dotenv()

app = create_app()
logging.info("Sidemed AI - Server started")
