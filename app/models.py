from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func

from app.database import Base

# Pydantic
#    = API data

# SQLAlchemy
#    = Database data
# API request model
class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


# Database model
class Message(Base):

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)

    conversation_id = Column(String, index=True)

    role = Column(String)

    content = Column(Text)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

class Conversation(Base):

    __tablename__ = "conversations"

    id = Column(String, primary_key=True, index=True)

    title = Column(String)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )