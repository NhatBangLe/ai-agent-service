from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def greetings():
    return {"message": "Hello world!"}