from fastapi import FastAPI

from api.main import api_router

# Agent().run()

app = FastAPI()
app.include_router(api_router)

