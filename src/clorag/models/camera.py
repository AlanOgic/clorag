"""Camera compatibility data models."""

from __future__ import annotations

import re
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


# =============================================================================
# Canonical Values for Data Normalization
# =============================================================================

class ControlPort(str, Enum):
    """Canonical control port types."""

    RS422 = "RS-422"
    RS232 = "RS-232"
    RS485 = "RS-485"
    ETHERNET = "Ethernet"
    USB = "USB"
    GPIO = "GPIO"
    LANC = "LANC"
    SDI = "SDI"
    HDMI = "HDMI"
    SERIAL = "Serial"
    WIFI = "Wi-Fi"
    BLUETOOTH = "Bluetooth"


class ControlProtocol(str, Enum):
    """Canonical control protocol types."""

    VISCA = "VISCA"
    VISCA_OVER_IP = "VISCA over IP"
    SONY_RCP = "Sony RCP"
    SONY_SIMPLE_IP = "Sony Simple IP"
    PANASONIC_AW = "Panasonic AW"
    PANASONIC_PTZ = "Panasonic PTZ"
    CANON_XC = "Canon XC"
    CANON_RC_IP = "Canon RC-IP"
    BLACKMAGIC_SDI = "Blackmagic SDI"
    BLACKMAGIC_IP = "Blackmagic IP"
    ARRI_ECS = "ARRI ECS"
    LANC = "LANC"
    PELCO_D = "Pelco-D"
    PELCO_P = "Pelco-P"
    NDI = "NDI"
    SRT = "SRT"
    CGI = "CGI"
    HTTP_API = "HTTP API"
    REST_API = "REST API"


class SupportedControl(str, Enum):
    """Canonical supported control types."""

    IRIS = "Iris"
    GAIN = "Gain"
    SHUTTER = "Shutter"
    WHITE_BALANCE = "White Balance"
    ND_FILTER = "ND Filter"
    FOCUS = "Focus"
    ZOOM = "Zoom"
    ISO = "ISO"
    APERTURE = "Aperture"
    GAMMA = "Gamma"
    COLOR = "Color"
    KNEE = "Knee"
    DETAIL = "Detail"
    BLACK_LEVEL = "Black Level"
    MASTER_BLACK = "Master Black"
    PAN = "Pan"
    TILT = "Tilt"
    PTZ = "PTZ"
    PRESET = "Preset"
    TALLY = "Tally"
    REC_START_STOP = "Rec Start/Stop"
    TIMECODE = "Timecode"
    BARS = "Bars"


# Normalization mappings (variations → canonical form)
PORT_ALIASES: dict[str, str] = {
    "rs-422": ControlPort.RS422.value,
    "rs422": ControlPort.RS422.value,
    "rs 422": ControlPort.RS422.value,
    "rs-232": ControlPort.RS232.value,
    "rs232": ControlPort.RS232.value,
    "rs 232": ControlPort.RS232.value,
    "rs-485": ControlPort.RS485.value,
    "rs485": ControlPort.RS485.value,
    "ethernet": ControlPort.ETHERNET.value,
    "eth": ControlPort.ETHERNET.value,
    "rj45": ControlPort.ETHERNET.value,
    "rj-45": ControlPort.ETHERNET.value,
    "lan": ControlPort.ETHERNET.value,
    "usb": ControlPort.USB.value,
    "usb-c": ControlPort.USB.value,
    "usb type-c": ControlPort.USB.value,
    "gpio": ControlPort.GPIO.value,
    "lanc": ControlPort.LANC.value,
    "sdi": ControlPort.SDI.value,
    "hd-sdi": ControlPort.SDI.value,
    "3g-sdi": ControlPort.SDI.value,
    "12g-sdi": ControlPort.SDI.value,
    "hdmi": ControlPort.HDMI.value,
    "serial": ControlPort.SERIAL.value,
    "wifi": ControlPort.WIFI.value,
    "wi-fi": ControlPort.WIFI.value,
    "wireless": ControlPort.WIFI.value,
    "bluetooth": ControlPort.BLUETOOTH.value,
    "bt": ControlPort.BLUETOOTH.value,
}

PROTOCOL_ALIASES: dict[str, str] = {
    "visca": ControlProtocol.VISCA.value,
    "visca/ip": ControlProtocol.VISCA_OVER_IP.value,
    "visca over ip": ControlProtocol.VISCA_OVER_IP.value,
    "visca-over-ip": ControlProtocol.VISCA_OVER_IP.value,
    "sony rcp": ControlProtocol.SONY_RCP.value,
    "sonyrcp": ControlProtocol.SONY_RCP.value,
    "sony simple ip": ControlProtocol.SONY_SIMPLE_IP.value,
    "panasonic aw": ControlProtocol.PANASONIC_AW.value,
    "panasonic ptz": ControlProtocol.PANASONIC_PTZ.value,
    "aw protocol": ControlProtocol.PANASONIC_AW.value,
    "canon xc": ControlProtocol.CANON_XC.value,
    "canon rc-ip": ControlProtocol.CANON_RC_IP.value,
    "canon rcip": ControlProtocol.CANON_RC_IP.value,
    "blackmagic sdi": ControlProtocol.BLACKMAGIC_SDI.value,
    "bmd sdi": ControlProtocol.BLACKMAGIC_SDI.value,
    "blackmagic ip": ControlProtocol.BLACKMAGIC_IP.value,
    "arri ecs": ControlProtocol.ARRI_ECS.value,
    "lanc": ControlProtocol.LANC.value,
    "pelco-d": ControlProtocol.PELCO_D.value,
    "pelco d": ControlProtocol.PELCO_D.value,
    "pelcod": ControlProtocol.PELCO_D.value,
    "pelco-p": ControlProtocol.PELCO_P.value,
    "pelco p": ControlProtocol.PELCO_P.value,
    "ndi": ControlProtocol.NDI.value,
    "srt": ControlProtocol.SRT.value,
    "cgi": ControlProtocol.CGI.value,
    "http api": ControlProtocol.HTTP_API.value,
    "http": ControlProtocol.HTTP_API.value,
    "rest api": ControlProtocol.REST_API.value,
    "rest": ControlProtocol.REST_API.value,
    "ip": ControlProtocol.HTTP_API.value,
}

CONTROL_ALIASES: dict[str, str] = {
    "iris": SupportedControl.IRIS.value,
    "gain": SupportedControl.GAIN.value,
    "shutter": SupportedControl.SHUTTER.value,
    "shutter speed": SupportedControl.SHUTTER.value,
    "white balance": SupportedControl.WHITE_BALANCE.value,
    "whitebalance": SupportedControl.WHITE_BALANCE.value,
    "wb": SupportedControl.WHITE_BALANCE.value,
    "nd filter": SupportedControl.ND_FILTER.value,
    "nd": SupportedControl.ND_FILTER.value,
    "neutral density": SupportedControl.ND_FILTER.value,
    "focus": SupportedControl.FOCUS.value,
    "auto focus": SupportedControl.FOCUS.value,
    "af": SupportedControl.FOCUS.value,
    "zoom": SupportedControl.ZOOM.value,
    "iso": SupportedControl.ISO.value,
    "aperture": SupportedControl.APERTURE.value,
    "f-stop": SupportedControl.APERTURE.value,
    "gamma": SupportedControl.GAMMA.value,
    "color": SupportedControl.COLOR.value,
    "colour": SupportedControl.COLOR.value,
    "color temperature": SupportedControl.COLOR.value,
    "knee": SupportedControl.KNEE.value,
    "detail": SupportedControl.DETAIL.value,
    "black level": SupportedControl.BLACK_LEVEL.value,
    "black": SupportedControl.BLACK_LEVEL.value,
    "master black": SupportedControl.MASTER_BLACK.value,
    "pan": SupportedControl.PAN.value,
    "tilt": SupportedControl.TILT.value,
    "ptz": SupportedControl.PTZ.value,
    "pan/tilt/zoom": SupportedControl.PTZ.value,
    "preset": SupportedControl.PRESET.value,
    "presets": SupportedControl.PRESET.value,
    "tally": SupportedControl.TALLY.value,
    "tally light": SupportedControl.TALLY.value,
    "rec start/stop": SupportedControl.REC_START_STOP.value,
    "record": SupportedControl.REC_START_STOP.value,
    "recording": SupportedControl.REC_START_STOP.value,
    "rec": SupportedControl.REC_START_STOP.value,
    "timecode": SupportedControl.TIMECODE.value,
    "tc": SupportedControl.TIMECODE.value,
    "bars": SupportedControl.BARS.value,
    "color bars": SupportedControl.BARS.value,
}


def normalize_port(port: str) -> str:
    """Normalize a port name to canonical form."""
    key = port.lower().strip()
    return PORT_ALIASES.get(key, port.strip())


def normalize_protocol(protocol: str) -> str:
    """Normalize a protocol name to canonical form."""
    key = protocol.lower().strip()
    return PROTOCOL_ALIASES.get(key, protocol.strip())


def normalize_control(control: str) -> str:
    """Normalize a control name to canonical form."""
    key = control.lower().strip()
    return CONTROL_ALIASES.get(key, control.strip())


def normalize_ports(ports: list[str]) -> list[str]:
    """Normalize a list of ports, removing duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for port in ports:
        normalized = normalize_port(port)
        if normalized.lower() not in seen:
            seen.add(normalized.lower())
            result.append(normalized)
    return result


def normalize_protocols(protocols: list[str]) -> list[str]:
    """Normalize a list of protocols, removing duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for protocol in protocols:
        normalized = normalize_protocol(protocol)
        if normalized.lower() not in seen:
            seen.add(normalized.lower())
            result.append(normalized)
    return result


def normalize_controls(controls: list[str]) -> list[str]:
    """Normalize a list of controls, removing duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for control in controls:
        normalized = normalize_control(control)
        if normalized.lower() not in seen:
            seen.add(normalized.lower())
            result.append(normalized)
    return result


# =============================================================================
# Device Type Inference
# =============================================================================

# Patterns for inferring device type from model name
DEVICE_TYPE_PATTERNS: dict[DeviceType, list[str]] = {
    DeviceType.CAMERA_PTZ: [
        r"\bPTZ\b",
        r"\bAW-[A-Z]{2}\d",  # Panasonic AW-HE/UE series
        r"\bBRC-",  # Sony BRC series
        r"\bPXW-Z\d+[A-Z]?$",  # Sony PXW-Z PTZ
        r"\bEVI-",  # Sony EVI series
        r"\bCR-N\d",  # Canon CR-N series
        r"\bVC-",  # Lumens/PTZOptics VC series
        r"\bPT\d",  # Various PT prefix
        r"\bMove\b",  # PTZOptics Move series
        r"\bRobo\b",  # Robotic cameras
    ],
    DeviceType.CAMERA_CINEMA: [
        r"\bFX\d",  # Sony FX series
        r"\bFX\s?\d",
        r"\bVENICE\b",  # Sony VENICE
        r"\bC\d{2,3}\b",  # Canon C200, C300, C500, C70
        r"\bC\d{2,3}\s+Mark",
        r"\bURSA\b",  # Blackmagic URSA
        r"\bPocket\b.*\d[Kk]",  # BMPCC
        r"\bALEXA\b",  # ARRI ALEXA
        r"\bAMIRA\b",  # ARRI AMIRA
        r"\bKOMODO\b",  # RED KOMODO
        r"\bRAPTOR\b",  # RED RAPTOR
        r"\bV-RAPTOR\b",
        r"\bMONSTRO\b",  # RED MONSTRO
        r"\bGEMINI\b",  # RED GEMINI
        r"\bMavo\b",  # Kinefinity
        r"\bTerra\b",  # Kinefinity
        r"\bE2\b",  # Z CAM E2
        r"\bVariCam\b",  # Panasonic VariCam
        r"\bAU-EVA\d",  # Panasonic EVA
    ],
    DeviceType.CAMERA_BROADCAST: [
        r"\bHDC-",  # Sony HDC series
        r"\bHXC-",  # Sony HXC series
        r"\bHDK-",  # Ikegami HDK
        r"\bHC-",  # Ikegami HC
        r"\bUHK-",  # Ikegami UHK
        r"\bLDX\b",  # Grass Valley LDX
        r"\bLDK\b",  # Grass Valley LDK
        r"\bHPX-",  # Panasonic HPX
        r"\bAK-",  # Panasonic AK
        r"\bGY-",  # JVC GY
        r"\bSK-",  # Hitachi SK
        r"\bZ-HD\d",  # Hitachi Z-HD
        r"\bXF\d{3}",  # Canon XF series
    ],
    DeviceType.CAMERA_MIRRORLESS: [
        r"\b[Aa]lpha\b",  # Sony Alpha
        r"\bα\d",  # Sony α
        r"\bA7\b",  # Sony A7
        r"\bA9\b",  # Sony A9
        r"\bILCE-",  # Sony ILCE model codes
        r"\bEOS\s?R\d*\b",  # Canon EOS R
        r"\bEOS\s?[RMC]\d",
        r"\bLUMIX\b",  # Panasonic Lumix
        r"\bDC-[SGB]",  # Panasonic DC series
        r"\bGH\d",  # Panasonic GH
        r"\bS1[HR]?\b",  # Panasonic S1
        r"\bS5\b",  # Panasonic S5
        r"\bZ\s?\d\b",  # Nikon Z
        r"\bX-[HTW]\d",  # Fujifilm X series
        r"\bGFX\b",  # Fujifilm GFX
    ],
    DeviceType.CAMERA_BOX: [
        r"\bBlock\b",  # Sony Block cameras
        r"\bFCB-",  # Sony FCB modules
        r"\bEV\d{3,4}",  # Sony EV series
        r"\bNCG-",
        r"\bBox\b",
        r"\bMini\b.*\bPro\b",  # Blackmagic Mini Pro
    ],
    DeviceType.CAMERA_ACTION: [
        r"\bGoPro\b",
        r"\bHERO\b",
        r"\bGP\d",
        r"\bInsta360\b",
        r"\bOsmo\s?Action\b",
        r"\bDJI\s?Action\b",
    ],
    DeviceType.CAMERA_MODULE: [
        r"\bFCB-",  # Sony FCB camera blocks
        r"\bEV\d{3,4}[A-Z]?\b",
        r"\bModule\b",
        r"\bBlock\b",
    ],
    DeviceType.LENS: [
        r"\b\d{1,3}-\d{2,4}mm\b",  # Focal length range
        r"\bCN\d{1,2}x",  # Canon CN-E lenses
        r"\bHJ\d{2}",  # Canon HJ broadcast lenses
        r"\bKJ\d{2}",  # Canon KJ lenses
        r"\bUA\d{2,3}",  # Fujinon UA lenses
        r"\bXA\d{2}",  # Fujinon XA lenses
        r"\bHA\d{2}",  # Fujinon HA lenses
        r"\bZK\d",  # Fujinon ZK series
        r"\bCabrio\b",  # Fujinon Cabrio
        r"\bAngénieux\b",
        r"\bAngenieux\b",
        r"\bOptimo\b",
        r"\bSignature\b.*\bPrime\b",
        r"\bSupreme\b.*\bPrime\b",
        r"\bMaster\b.*\bPrime\b",
        r"\bUltra\b.*\bPrime\b",
    ],
    DeviceType.LENS_MOTOR: [
        r"\bCmotion\b",
        r"\bFIZ\b",
        r"\bWCU-",  # ARRI WCU
        r"\bSXU-",  # ARRI SXU
        r"\bNucleus\b",  # Tilta Nucleus
        r"\bFollow\s?Focus\b",
        r"\bLens\s?Motor\b",
        r"\bPreston\b",
        r"\bHeden\b",
    ],
    DeviceType.SWITCHER: [
        r"\bATEM\b",  # Blackmagic ATEM
        r"\bSwitch\w*\b",
        r"\bMixer\b",
        r"\bV-\d+HD\b",  # Roland V-series
        r"\bAWS-",  # Sony AWS
    ],
    DeviceType.GIMBAL: [
        r"\bGimbal\b",
        r"\bRonin\b",  # DJI Ronin
        r"\bRS\s?\d\b",  # DJI RS
        r"\bMovi\b",  # Freefly Movi
        r"\bStabilizer\b",
        r"\bWeebill\b",  # Zhiyun Weebill
        r"\bCrane\b",  # Zhiyun Crane
    ],
    DeviceType.HEAD: [
        r"\bHead\b",
        r"\bPan.*Tilt\b",
        r"\bO'Connor\b",
        r"\bSachtler\b",
        r"\bVinten\b",
        r"\bCartoni\b",
        r"\bFluid\s?Head\b",
    ],
    DeviceType.ENCODER: [
        r"\bEncoder\b",
        r"\bStreamer\b",
        r"\bWeb\s?Presenter\b",
        r"\bAJA\s?HELO\b",
        r"\bKiloview\b",
        r"\bMagewell\b",
    ],
}


def infer_device_type(name: str, manufacturer: str | None = None) -> DeviceType | None:
    """Infer device type from model name and manufacturer.

    Args:
        name: Camera/device model name.
        manufacturer: Manufacturer name (optional, for context).

    Returns:
        Inferred DeviceType or None if uncertain.
    """
    # Combine name and manufacturer for matching
    search_text = name
    if manufacturer:
        search_text = f"{manufacturer} {name}"

    # Check each device type's patterns
    for device_type, patterns in DEVICE_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, search_text, re.IGNORECASE):
                return device_type

    return None


class Camera(BaseModel):
    """Camera compatibility information."""

    id: int | None = None
    name: str = Field(..., description="Camera model name (e.g., 'HDC-5500')")
    manufacturer: str | None = Field(None, description="Camera manufacturer (e.g., 'Sony')")
    code_model: str | None = Field(
        None, description="Official model code from manufacturer (e.g., 'ILME-FX6V')",
    )
    device_type: DeviceType | None = Field(
        None, description="Device category (camera_cinema, camera_ptz, lens, etc.)",
    )
    ports: list[str] = Field(
        default_factory=list, description="Control ports (RS-422, Ethernet, etc.)",
    )
    protocols: list[str] = Field(
        default_factory=list, description="Control protocols (VISCA, Sony RCP, etc.)",
    )
    supported_controls: list[str] = Field(
        default_factory=list, description="Supported controls (Iris, Gain, Shutter, etc.)"
    )
    notes: list[str] = Field(default_factory=list, description="Important notes and requirements")
    source: CameraSource = Field(CameraSource.MANUAL, description="Source of the information")
    doc_url: str | None = Field(None, description="Link to documentation page")
    manufacturer_url: str | None = Field(None, description="Link to manufacturer product page")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Extraction confidence score (0-1)",
    )
    needs_review: bool = Field(default=False, description="Flag for human review")
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
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    needs_review: bool = Field(default=False)


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
    confidence: float | None = None
    needs_review: bool | None = None


class CameraEnrichment(BaseModel):
    """Data enriched from manufacturer website."""

    specs: dict[str, str] = Field(default_factory=dict, description="Technical specifications")
    features: list[str] = Field(default_factory=list, description="Product features")
    connectivity: list[str] = Field(default_factory=list, description="Connectivity options")
    remote_control: list[str] = Field(
        default_factory=list, description="Remote control capabilities",
    )
    source_url: str | None = None


# =============================================================================
# Extraction Validation
# =============================================================================

# Known manufacturers for validation
KNOWN_MANUFACTURERS: set[str] = {
    "Sony", "Canon", "Panasonic", "Blackmagic", "ARRI", "RED", "Grass Valley",
    "Ikegami", "Hitachi", "JVC", "Fujifilm", "Nikon", "Z CAM", "Kinefinity",
    "Ross", "AJA", "GoPro", "DJI", "Atomos", "Marshall", "PTZOptics", "Lumens",
    "BirdDog", "Magewell", "Kiloview", "Datavideo", "Roland", "NewTek",
    "Tilta", "DZOFilm", "Sigma", "Zeiss", "Cooke", "Angénieux", "Fujinon",
    "Preston", "ARRI", "Teradek", "Hollyland", "Zhiyun", "Freefly",
}

# Phrases that indicate hallucination or generic mentions
HALLUCINATION_INDICATORS: set[str] = {
    "any camera", "most cameras", "all cameras", "example camera",
    "generic camera", "various cameras", "multiple cameras",
    "camera model", "your camera", "the camera", "a camera",
}


class ValidationResult(BaseModel):
    """Result of camera extraction validation."""

    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    needs_review: bool = False


def validate_camera_extraction(camera: CameraCreate) -> ValidationResult:
    """Validate extracted camera data and compute confidence score.

    Args:
        camera: Camera data extracted by LLM.

    Returns:
        ValidationResult with validity, confidence, and any issues.
    """
    issues: list[str] = []
    confidence = 1.0

    # Check model name
    name_lower = camera.name.lower()

    # Rule 1: Name length (too short or too long)
    if len(camera.name) < 2:
        issues.append(f"Model name too short: '{camera.name}'")
        confidence -= 0.4
    elif len(camera.name) > 50:
        issues.append(f"Model name suspiciously long: '{camera.name}'")
        confidence -= 0.2

    # Rule 2: Hallucination indicators
    for indicator in HALLUCINATION_INDICATORS:
        if indicator in name_lower:
            issues.append(f"Likely hallucination - name contains '{indicator}'")
            confidence -= 0.5
            break

    # Rule 3: Manufacturer validation
    if camera.manufacturer:
        # Normalize for comparison
        mfr_normalized = camera.manufacturer.lower().replace(" ", "")
        known_normalized = {m.lower().replace(" ", "") for m in KNOWN_MANUFACTURERS}

        if mfr_normalized not in known_normalized:
            issues.append(f"Unknown manufacturer: '{camera.manufacturer}'")
            confidence -= 0.15
    else:
        issues.append("No manufacturer specified")
        confidence -= 0.1

    # Rule 4: Must have some compatibility info
    has_ports = len(camera.ports) > 0
    has_protocols = len(camera.protocols) > 0
    has_controls = len(camera.supported_controls) > 0

    if not has_ports and not has_protocols and not has_controls:
        issues.append("No compatibility info (ports, protocols, or controls)")
        confidence -= 0.3

    # Rule 5: Suspicious patterns in name
    suspicious_patterns = [
        r"^\d+$",  # Just numbers
        r"^[a-z]$",  # Single letter
        r"test",  # Test entries
        r"sample",
        r"example",
        r"unknown",
        r"n/a",
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, name_lower):
            issues.append(f"Suspicious pattern in name: '{camera.name}'")
            confidence -= 0.3
            break

    # Rule 6: Name contains manufacturer (should be cleaned)
    if camera.manufacturer:
        if name_lower.startswith(camera.manufacturer.lower()):
            issues.append("Model name includes manufacturer prefix")
            confidence -= 0.1

    # Clamp confidence
    confidence = max(0.0, min(1.0, confidence))

    # Determine if needs review
    needs_review = confidence < 0.7 or len(issues) > 0

    return ValidationResult(
        is_valid=confidence >= 0.5 and not any("hallucination" in i.lower() for i in issues),
        confidence=confidence,
        issues=issues,
        needs_review=needs_review,
    )


def normalize_camera_create(camera: CameraCreate) -> CameraCreate:
    """Normalize a CameraCreate model's ports, protocols, and controls.

    Also infers device_type if not set.

    Args:
        camera: Original camera data.

    Returns:
        New CameraCreate with normalized values.
    """
    # Infer device type if not set
    device_type = camera.device_type
    if device_type is None:
        device_type = infer_device_type(camera.name, camera.manufacturer)

    return CameraCreate(
        name=camera.name,
        manufacturer=camera.manufacturer,
        code_model=camera.code_model,
        device_type=device_type,
        ports=normalize_ports(camera.ports),
        protocols=normalize_protocols(camera.protocols),
        supported_controls=normalize_controls(camera.supported_controls),
        notes=camera.notes,
        doc_url=camera.doc_url,
        manufacturer_url=camera.manufacturer_url,
        confidence=camera.confidence,
        needs_review=camera.needs_review,
    )
