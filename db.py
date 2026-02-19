from sqlalchemy import (create_engine, Column, Integer, String, Float, DateTime,
                        ForeignKey, JSON)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timezone

DB_URL = "sqlite:///train_counter.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class TrainPass(Base):
    __tablename__ = "train_pass"
    id = Column(Integer, primary_key=True)
    train_id = Column(String, unique=True, index=True)
    start_ts = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    end_ts = Column(DateTime, nullable=True)
    direction = Column(String)  # "L->R", "R->L", or None
    total_locomotives = Column(Integer, default=0)
    total_railcars = Column(Integer, default=0)
    avg_speed_mph = Column(Float, nullable=True)
    extra = Column(JSON, default={})  # room for future metadata

    events = relationship("CarEvent", back_populates="train_pass", cascade="all,delete-orphan")
    engines = relationship("EngineSighting", back_populates="train_pass", cascade="all,delete-orphan")

class CarEvent(Base):
    __tablename__ = "car_event"
    id = Column(Integer, primary_key=True)
    train_pass_id = Column(Integer, ForeignKey("train_pass.id"))
    track_id = Column(Integer)
    klass = Column(String)      # "locomotive" or "railcar"
    crossed_ts = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    direction = Column(String)  # "L->R" or "R->L"

    train_pass = relationship("TrainPass", back_populates="events")

class EngineSighting(Base):
    __tablename__ = "engine_sighting"
    id = Column(Integer, primary_key=True)
    train_pass_id = Column(Integer, ForeignKey("train_pass.id"))
    track_id = Column(Integer)
    engine_number = Column(String, index=True)  # "####"
    first_seen_ts = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    train_pass = relationship("TrainPass", back_populates="engines")

def init_db():
    Base.metadata.create_all(engine)