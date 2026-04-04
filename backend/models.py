from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=True)  # null for Kakao users
    kakao_id = Column(String, unique=True, index=True, nullable=True)
    google_id = Column(String, unique=True, index=True, nullable=True)
    auth_provider = Column(String, default="email")  # "email", "kakao", "google"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    purchases = relationship("Purchase", back_populates="user")
    reviews = relationship("Review", back_populates="user")


class Purchase(Base):
    __tablename__ = "purchases"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    plan_name = Column(String, nullable=False)  # basic, plus, pro
    price = Column(Integer, nullable=False)
    items_remaining = Column(Integer, nullable=False)   # 문항 첨삭 남은 횟수
    revisions_remaining = Column(Integer, nullable=False)  # 수정 남은 횟수
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="purchases")


class PaymentOrder(Base):
    __tablename__ = "payment_orders"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    plan_name = Column(String, nullable=False)
    amount = Column(Integer, nullable=False)
    status = Column(String, default="pending")  # pending, paid, failed
    payment_key = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    purchase_id = Column(Integer, ForeignKey("purchases.id"), nullable=False)
    company_name = Column(String, nullable=False)
    question = Column(Text, nullable=False)
    original_text = Column(Text, nullable=False)
    result = Column(Text, nullable=True)
    revision_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="reviews")
