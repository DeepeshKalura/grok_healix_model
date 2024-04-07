from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.emo import get_response, langchain_get_response


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return FileResponse("template/root.html")


class Message(BaseModel):
    message: str


@app.post("/pranjal_promot")
def support(message: Message):
    return {"response":  langchain_get_response(message.message)}

@app.post("/deepesh_promot")
def provide_support(message: Message):
    return {"response":  get_response(message.message)}
