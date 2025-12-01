"""Camera compatibility data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class CameraSource(str, Enum):
    """Source of camera information."""

    DOCUMENTATION = "documentation"
    SUPPORT_CASE = "support_case"
    MANUFACTURER = "manufacturer"
    MANUAL = "manual"


class DeviceType(str, Enum):
    """Type/category of device."""

    CAMERA_CINEMA = "camera_cinema"
    CAMERA_BROADCAST = "camera_broadcast"
    CAMERA_PTZ = "camera_ptz"
    CAMERA_MIRRORLESS = "camera_mirrorless"
    CAMERA_BOX = "camera_box"
    CAMERA_ACTION = "camera_action"
    CAMERA_MODULE = "camera_module"  # FCB blocks, etc.
    LENS = "lens"
    LENS_MOTOR = "lens_motor"
    SWITCHER = "switcher"
    GIMBAL = "gimbal"
    HEAD = "head"  # Pan/tilt heads
    ENCODER = "encoder"
    OTHER = "other"


class Camera(BaseModel):
    """Camera compatibility information."""

    id: int | None = None
    name: str = Field(..., description="Camera model name (e.g., 'HDC-5500')")
    manufacturer: str | None = Field(None, description="Camera manufacturer (e.g., 'Sony')")
    code_model: str | None = Field(None, description="Official model code from manufacturer (e.g., 'ILME-FX6V')")
    device_type: DeviceType | None = Field(None, description="Device category (camera_cinema, camera_ptz, lens, etc.)")
    ports: list[str] = Field(default_factory=list, description="Control ports (RS-422, Ethernet, etc.)")
    protocols: list[str] = Field(default_factory=list, description="Control protocols (VISCA, Sony RCP, etc.)")
    supported_controls: list[str] = Field(
        default_factory=list, description="Supported controls (Iris, Gain, Shutter, etc.)"
    )
    notes: list[str] = Field(default_factory=list, description="Important notes and requirements")
    source: CameraSource = Field(CameraSource.MANUAL, description="Source of the information")
    doc_url: str | None = Field(None, description="Link to documentation page")
    manufacturer_url: str | None = Field(None, description="Link to manufacturer product page")
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CameraCreate(BaseModel):
    """Model for creating a new camera entry."""

    name: str
    manufacturer: str | None = None
    code_model: str | None = None
    device_type: DeviceType | None = None
    ports: list[str] = Field(default_factory=list)
    protocols: list[str] = Field(default_factory=list)
    supported_controls: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    doc_url: str | None = None
    manufacturer_url: str | None = None


class CameraUpdate(BaseModel):
    """Model for updating an existing camera entry."""

    name: str | None = None
    manufacturer: str | None = None
    code_model: str | None = None
    device_type: DeviceType | None = None
    ports: list[str] | None = None
    protocols: list[str] | None = None
    supported_controls: list[str] | None = None
    notes: list[str] | None = None
    doc_url: str | None = None
    manufacturer_url: str | None = None


class CameraEnrichment(BaseModel):
    """Data enriched from manufacturer website."""

    specs: dict[str, str] = Field(default_factory=dict, description="Technical specifications")
    features: list[str] = Field(default_factory=list, description="Product features")
    connectivity: list[str] = Field(default_factory=list, description="Connectivity options")
    remote_control: list[str] = Field(default_factory=list, description="Remote control capabilities")
    source_url: str | None = None
