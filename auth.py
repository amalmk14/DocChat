import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from models import User
from db import get_db

SECRET = os.environ["SECRET_KEY"]
router = APIRouter(prefix="/auth")
bearer = HTTPBearer()


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)


def make_token(user_id: int) -> str:
    expires = datetime.now(timezone.utc) + timedelta(days=1)
    return jwt.encode({"sub": str(user_id), "exp": expires}, SECRET, algorithm="HS256")


def current_user(creds: HTTPAuthorizationCredentials = Depends(bearer), db: Session = Depends(get_db)):
    try:
        user_id = int(jwt.decode(creds.credentials, SECRET, algorithms=["HS256"])["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(401, "Invalid or expired token")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(401, "User not found")
    return user


@router.post("/register")
def register(body: Credentials, db: Session = Depends(get_db)):
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "Email already registered")
    hashed = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
    user = User(email=email, password=hashed)
    db.add(user)
    db.commit()
    return {"token": make_token(user.id)}


@router.post("/login")
def login(body: Credentials, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if not user or not bcrypt.checkpw(body.password.encode(), user.password.encode()):
        raise HTTPException(401, "Wrong email or password")
    return {"token": make_token(user.id)}