from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class Error(BaseModel):
    detail: str


class LoginInput(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class UserView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    email: str


class VersionView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    number: int
    filename: str
    sha256: str
    size_bytes: int
    row_count: int
    columns: list[str]
    created_at: datetime


class DatasetView(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    version: VersionView


class DatasetList(BaseModel):
    items: list[DatasetView]
    total: int
    page: int
    page_size: int


class Preview(BaseModel):
    version_id: UUID
    columns: list[str]
    rows: list[list[str]]
    total_rows: int
    page: int
    page_size: int
