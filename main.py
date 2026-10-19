import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

import auth
import rag
from db import SessionLocal, get_db, init_db
from models import Document, Message

MAX_MB = 10
NO_DOCS = "I couldn't find any documents. Please upload a PDF first."


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="DocChat", lifespan=lifespan)
app.include_router(auth.router)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "templates" / "index.html")


# ---------- documents ----------
@app.post("/documents")
def upload(file: UploadFile = File(...), db: Session = Depends(get_db), user=Depends(auth.current_user)):
    name = os.path.basename(file.filename or "")
    data = file.file.read(MAX_MB * 1024 * 1024 + 1)
    if not name.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
        raise HTTPException(400, "Please upload a valid PDF")
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File is bigger than {MAX_MB} MB")
    try:
        chunks = rag.save_pdf(db, user.id, name, data)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"chunks": chunks}


@app.get("/documents")
def list_documents(db: Session = Depends(get_db), user=Depends(auth.current_user)):
    docs = db.scalars(select(Document).where(Document.user_id == user.id).order_by(Document.id.desc()))
    return [{"id": d.id, "filename": d.filename} for d in docs]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db), user=Depends(auth.current_user)):
    doc = db.get(Document, doc_id)
    if not doc or doc.user_id != user.id:
        raise HTTPException(404, "Document not found")
    db.delete(doc)  
    db.commit()
    return {"ok": True}


# ---------- chat history ----------
@app.get("/history")
def get_history(db: Session = Depends(get_db), user=Depends(auth.current_user)):
    rows = db.scalars(select(Message).where(Message.user_id == user.id).order_by(Message.id))
    return [{"role": m.role, "content": m.content, "sources": m.sources} for m in rows]


@app.delete("/history")
def clear_history(db: Session = Depends(get_db), user=Depends(auth.current_user)):
    db.execute(delete(Message).where(Message.user_id == user.id))
    db.commit()
    return {"ok": True}


# ---------- chat ----------
class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"  

@app.post("/chat")
def chat(body: Ask, db: Session = Depends(get_db), user=Depends(auth.current_user)):
    recent = db.scalars(
        select(Message).where(Message.user_id == user.id).order_by(Message.id.desc()).limit(6)
    ).all()
    history = [{"role": m.role, "content": m.content} for m in reversed(recent)]
    while history and history[0]["role"] != "user":
        history.pop(0)

    previous_question = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    chunks = rag.get_context(db, user.id, body.question, previous_question)

    db.add(Message(user_id=user.id, role="user", content=body.question))
    db.commit()
    user_id, question = user.id, body.question

    def events():
        parts = []
        try:
            if chunks:
                for token in rag.stream_answer(history, question, chunks):
                    parts.append(token)
                    yield sse({"token": token})
            else:
                parts.append(NO_DOCS)
                yield sse({"token": NO_DOCS})
        except Exception:
            logging.exception("Gemini call failed")
            yield sse({"error": "The AI service failed. Please try again."})
            return

        with SessionLocal() as session:
            session.add(Message(user_id=user_id, role="assistant", content="".join(parts)))
            session.commit()
        yield sse({"done": True})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})