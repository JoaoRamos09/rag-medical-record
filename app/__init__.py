from fastapi import FastAPI
from app.controller.medical_records_controller import medical_records_router
import logging
import sys
import coloredlogs

def create_app():
    
    coloredlogs.install(level='INFO', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout, level_styles={
        'info': {'color': 'green'},
        'warning': {'color': 'yellow'},
        'error': {'color': 'red', 'bold': True},
        'critical': {'color': 'red', 'bold': True}
    })
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("primp").setLevel(logging.WARNING)
    
    app = FastAPI()
    app.include_router(medical_records_router)
    return app