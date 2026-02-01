"""Async HTTP client for Odoo MCP Server integration.

This client communicates with the Odoo MCP Server (FastMCP streamable-http)
to access Odoo ERP functionality for CRM, sales, and support operations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import httpx
import structlog

from clorag.config import get_settings
from clorag.core.cache import LRUCache, make_cache_key

logger = structlog.get_logger(__name__)


class OdooMCPError(Exception):
    """Base exception for Odoo MCP client errors."""

    pass


class OdooMCPConnectionError(OdooMCPError):
    """Connection error to Odoo MCP server."""

    pass


class OdooMCPAuthError(OdooMCPError):
    """Authentication error with Odoo MCP server."""

    pass


class OdooMCPAPIError(OdooMCPError):
    """API error returned by Odoo MCP server."""

    def __init__(self, message: str, code: int | None = None, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


class PartnerType(str, Enum):
    """Odoo partner types."""

    CONTACT = "contact"
    COMPANY = "company"


class OrderState(str, Enum):
    """Odoo sale order states."""

    DRAFT = "draft"
    SENT = "sent"
    SALE = "sale"
    DONE = "done"
    CANCEL = "cancel"


@dataclass
class OdooCustomer:
    """Odoo customer (res.partner) data model."""

    id: int
    name: str
    email: str | None = None
    phone: str | None = None
    mobile: str | None = None
    company_name: str | None = None
    country: str | None = None
    country_id: int | None = None
    vat: str | None = None
    is_company: bool = False
    parent_id: int | None = None
    street: str | None = None
    city: str | None = None
    zip_code: str | None = None
    commercial_partner_id: int | None = None

    @classmethod
    def from_odoo(cls, data: dict[str, Any]) -> OdooCustomer:
        """Create from Odoo record data."""
        # Extract Many2one fields (can be [id, name] or just id)
        country_id_raw = data.get("country_id")
        parent_id_raw = data.get("parent_id")
        commercial_partner_raw = data.get("commercial_partner_id")

        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            email=data.get("email"),
            phone=data.get("phone"),
            mobile=data.get("mobile"),
            company_name=data.get("company_name") or (
                parent_id_raw[1] if isinstance(parent_id_raw, list) else None
            ),
            country=(
                country_id_raw[1] if isinstance(country_id_raw, list) else None
            ),
            country_id=(
                country_id_raw[0] if isinstance(country_id_raw, list)
                else country_id_raw
            ),
            vat=data.get("vat"),
            is_company=data.get("is_company", False),
            parent_id=(
                parent_id_raw[0] if isinstance(parent_id_raw, list)
                else parent_id_raw
            ),
            street=data.get("street"),
            city=data.get("city"),
            zip_code=data.get("zip"),
            commercial_partner_id=(
                commercial_partner_raw[0] if isinstance(commercial_partner_raw, list)
                else commercial_partner_raw
            ),
        )


@dataclass
class OdooContact:
    """Odoo contact (child of company partner)."""

    id: int
    name: str
    email: str | None = None
    phone: str | None = None
    mobile: str | None = None
    function: str | None = None  # Job title

    @classmethod
    def from_odoo(cls, data: dict[str, Any]) -> OdooContact:
        """Create from Odoo record data."""
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            email=data.get("email"),
            phone=data.get("phone"),
            mobile=data.get("mobile"),
            function=data.get("function"),
        )


@dataclass
class OdooProduct:
    """Odoo product data model."""

    id: int
    name: str
    default_code: str | None = None  # Internal reference / SKU
    list_price: float = 0.0
    categ_id: int | None = None
    categ_name: str | None = None
    type: str = "consu"  # consu, service, product
    description: str | None = None
    active: bool = True

    @classmethod
    def from_odoo(cls, data: dict[str, Any]) -> OdooProduct:
        """Create from Odoo record data."""
        categ = data.get("categ_id")
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            default_code=data.get("default_code"),
            list_price=data.get("list_price", 0.0),
            categ_id=categ[0] if isinstance(categ, list) else categ,
            categ_name=categ[1] if isinstance(categ, list) else None,
            type=data.get("type", "consu"),
            description=data.get("description_sale") or data.get("description"),
            active=data.get("active", True),
        )


@dataclass
class OdooQuotationLine:
    """Odoo quotation/order line."""

    product_id: int
    name: str | None = None
    quantity: float = 1.0
    price_unit: float | None = None
    discount: float = 0.0


@dataclass
class OdooQuotation:
    """Odoo quotation (sale.order in draft/sent state)."""

    id: int
    name: str  # e.g., S00001
    partner_id: int
    partner_name: str | None = None
    state: OrderState = OrderState.DRAFT
    date_order: datetime | None = None
    amount_total: float = 0.0
    amount_untaxed: float = 0.0
    currency: str = "EUR"
    lines: list[dict[str, Any]] = field(default_factory=list)
    validity_date: datetime | None = None
    note: str | None = None

    @classmethod
    def from_odoo(cls, data: dict[str, Any]) -> OdooQuotation:
        """Create from Odoo record data."""
        partner = data.get("partner_id")
        currency = data.get("currency_id")
        date_order_raw = data.get("date_order")
        validity_raw = data.get("validity_date")

        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            partner_id=partner[0] if isinstance(partner, list) else partner or 0,
            partner_name=partner[1] if isinstance(partner, list) else None,
            state=OrderState(data.get("state", "draft")),
            date_order=(
                datetime.fromisoformat(date_order_raw) if date_order_raw else None
            ),
            amount_total=data.get("amount_total", 0.0),
            amount_untaxed=data.get("amount_untaxed", 0.0),
            currency=currency[1] if isinstance(currency, list) else "EUR",
            lines=data.get("order_line", []),
            validity_date=(
                datetime.fromisoformat(validity_raw) if validity_raw else None
            ),
            note=data.get("note"),
        )


@dataclass
class OdooPurchase:
    """Simplified purchase/sales history entry."""

    id: int
    name: str
    date: datetime | None
    state: str
    amount_total: float
    product_names: list[str] = field(default_factory=list)


@dataclass
class OdooRepair:
    """Odoo repair order (repair.order)."""

    id: int
    name: str
    partner_id: int
    partner_name: str | None = None
    product_id: int | None = None
    product_name: str | None = None
    lot_id: int | None = None
    lot_name: str | None = None  # Serial number
    state: str = "draft"
    description: str | None = None
    create_date: datetime | None = None

    @classmethod
    def from_odoo(cls, data: dict[str, Any]) -> OdooRepair:
        """Create from Odoo record data."""
        partner = data.get("partner_id")
        product = data.get("product_id")
        lot = data.get("lot_id")
        create_date_raw = data.get("create_date")

        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            partner_id=partner[0] if isinstance(partner, list) else partner or 0,
            partner_name=partner[1] if isinstance(partner, list) else None,
            product_id=product[0] if isinstance(product, list) else product,
            product_name=product[1] if isinstance(product, list) else None,
            lot_id=lot[0] if isinstance(lot, list) else lot,
            lot_name=lot[1] if isinstance(lot, list) else None,
            state=data.get("state", "draft"),
            description=data.get("description") or data.get("internal_notes"),
            create_date=(
                datetime.fromisoformat(create_date_raw) if create_date_raw else None
            ),
        )


@dataclass
class OdooSerial:
    """Odoo serial number (stock.lot) data model."""

    id: int
    name: str  # Serial number
    product_id: int | None = None
    product_name: str | None = None
    company_id: int | None = None
    company_name: str | None = None
    create_date: datetime | None = None
    # Delivery info (populated when tracing)
    customer_id: int | None = None
    customer_name: str | None = None
    delivery_date: datetime | None = None
    delivery_ref: str | None = None

    @classmethod
    def from_odoo(cls, data: dict[str, Any]) -> OdooSerial:
        """Create from Odoo record data."""
        product = data.get("product_id")
        company = data.get("company_id")
        create_date_raw = data.get("create_date")

        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            product_id=product[0] if isinstance(product, list) else product,
            product_name=product[1] if isinstance(product, list) else None,
            company_id=company[0] if isinstance(company, list) else company,
            company_name=company[1] if isinstance(company, list) else None,
            create_date=(
                datetime.fromisoformat(create_date_raw) if create_date_raw else None
            ),
        )


@dataclass
class WarrantyStatus:
    """Warranty status for a serial number."""

    serial: str
    product_name: str | None = None
    product_id: int | None = None
    purchase_date: datetime | None = None
    warranty_end: datetime | None = None
    is_under_warranty: bool = False
    customer_id: int | None = None
    customer_name: str | None = None


class OdooMCPClient:
    """Async client for Odoo MCP Server.

    Communicates with Odoo MCP Server using JSON-RPC 2.0 over HTTP.
    The MCP server acts as a bridge to Odoo's JSON-2 API.

    Performance optimizations:
    - Persistent HTTP client with connection pooling
    - LRU cache with TTL for read operations
    - Batch operations to reduce N+1 queries
    - Parallel enrichment via asyncio.gather()
    """

    # Fields to request for common operations
    PARTNER_FIELDS = [
        "id", "name", "email", "phone", "mobile", "company_name",
        "country_id", "vat", "is_company", "parent_id", "street",
        "city", "zip", "commercial_partner_id"
    ]

    PRODUCT_FIELDS = [
        "id", "name", "default_code", "list_price", "categ_id",
        "type", "description_sale", "active"
    ]

    ORDER_FIELDS = [
        "id", "name", "partner_id", "state", "date_order",
        "amount_total", "amount_untaxed", "currency_id", "order_line",
        "validity_date", "note"
    ]

    REPAIR_FIELDS = [
        "id", "name", "partner_id", "product_id", "lot_id",
        "state", "description", "internal_notes", "create_date"
    ]

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        cache_ttl: int | None = None,
    ):
        """Initialize the Odoo MCP client.

        Args:
            base_url: Odoo MCP server URL (defaults to settings).
            api_key: Bearer token for authentication (defaults to settings).
            timeout: Request timeout in seconds (defaults to settings).
            cache_ttl: Cache TTL in seconds for read operations (defaults to settings).
        """
        settings = get_settings()

        self._base_url = (base_url or settings.odoo_mcp_url).rstrip("/")
        self._api_key = api_key or (
            settings.odoo_mcp_api_key.get_secret_value()
            if settings.odoo_mcp_api_key else None
        )
        self._timeout = timeout or settings.odoo_mcp_timeout
        self._cache_ttl = cache_ttl or settings.odoo_mcp_cache_ttl

        # Cache for read operations
        self._cache: LRUCache[Any] = LRUCache(
            max_size=100, ttl_seconds=float(self._cache_ttl)
        )

        # Request ID counter for JSON-RPC
        self._request_id = 0

        # Persistent HTTP client with connection pooling
        # Reuses TCP connections across requests (~50-100ms savings per request)
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create the persistent HTTP client.

        Lazily initializes client with connection pooling for better performance.
        """
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=30.0,
                ),
            )
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client and release connections."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    def _get_next_request_id(self) -> int:
        """Get next JSON-RPC request ID."""
        self._request_id += 1
        return self._request_id

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for requests."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        use_cache: bool = False,
    ) -> Any:
        """Call an MCP tool on the Odoo MCP server.

        The FastMCP server exposes tools via the /mcp endpoint using
        JSON-RPC 2.0 protocol (tools/call method).

        Args:
            tool_name: Name of the MCP tool to call.
            arguments: Tool arguments.
            use_cache: Whether to cache the result.

        Returns:
            Tool result.

        Raises:
            OdooMCPError: On any error.
        """
        # Check cache for read operations
        cache_key = ""
        if use_cache:
            # Create cache key from tool name and arguments JSON
            import json as json_mod
            args_str = json_mod.dumps(arguments, sort_keys=True)
            cache_key = make_cache_key(tool_name, args_str)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for Odoo MCP call", tool=tool_name)
                return cached

        # Build JSON-RPC request for MCP tools/call
        request_data = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        url = f"{self._base_url}/mcp"

        try:
            # Use persistent HTTP client with connection pooling
            client = await self._get_http_client()
            response = await client.post(
                url,
                json=request_data,
                headers=self._get_headers(),
            )

            if response.status_code == 401:
                raise OdooMCPAuthError("Authentication failed with Odoo MCP server")

            if response.status_code == 403:
                raise OdooMCPAuthError("Access denied to Odoo MCP server")

            response.raise_for_status()

            result = response.json()

        except httpx.ConnectError as e:
            raise OdooMCPConnectionError(f"Cannot connect to Odoo MCP server at {url}: {e}") from e
        except httpx.TimeoutException as e:
            raise OdooMCPConnectionError(f"Timeout connecting to Odoo MCP server: {e}") from e
        except httpx.HTTPStatusError as e:
            raise OdooMCPAPIError(f"HTTP error from Odoo MCP server: {e}") from e

        # Handle JSON-RPC error response
        if "error" in result:
            error = result["error"]
            raise OdooMCPAPIError(
                message=error.get("message", "Unknown error"),
                code=error.get("code"),
                data=error.get("data"),
            )

        # Extract result from JSON-RPC response
        tool_result = result.get("result", {})

        # MCP tool results come as content array
        if isinstance(tool_result, dict) and "content" in tool_result:
            content = tool_result["content"]
            if content and isinstance(content, list) and len(content) > 0:
                # Get text content from first content item
                first_content = content[0]
                if isinstance(first_content, dict) and first_content.get("type") == "text":
                    text = first_content.get("text", "{}")
                    try:
                        tool_result = json.loads(text)
                    except json.JSONDecodeError:
                        tool_result = text

        # Check for error in tool result
        if isinstance(tool_result, dict) and tool_result.get("success") is False:
            raise OdooMCPAPIError(
                message=tool_result.get("error", "Tool execution failed"),
            )

        # Cache successful results
        if use_cache and cache_key:
            self._cache.set(cache_key, tool_result)

        return tool_result

    async def execute_method(
        self,
        model: str,
        method: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        use_cache: bool = False,
    ) -> Any:
        """Execute an arbitrary method on an Odoo model.

        This is the low-level method that maps to the execute_method tool
        on the Odoo MCP server.

        Args:
            model: Odoo model name (e.g., 'res.partner').
            method: Method name (e.g., 'search_read').
            args: Positional arguments as list.
            kwargs: Keyword arguments as dict.
            use_cache: Whether to cache the result.

        Returns:
            Method result from Odoo.
        """
        arguments: dict[str, Any] = {
            "model": model,
            "method": method,
        }

        if args:
            arguments["args_json"] = json.dumps(args)
        if kwargs:
            arguments["kwargs_json"] = json.dumps(kwargs)

        result = await self._call_tool("execute_method", arguments, use_cache=use_cache)

        # Extract actual result from execute_method response
        if isinstance(result, dict) and "result" in result:
            return result["result"]

        return result

    # =========================================================================
    # Customer Operations
    # =========================================================================

    async def lookup_customer_by_email(self, email: str) -> OdooCustomer | None:
        """Look up a customer by email address.

        Args:
            email: Customer email address.

        Returns:
            OdooCustomer or None if not found.
        """
        result = await self.execute_method(
            model="res.partner",
            method="search_read",
            kwargs={
                "domain": [["email", "=ilike", email]],
                "fields": self.PARTNER_FIELDS,
                "limit": 1,
            },
            use_cache=True,
        )

        if result and isinstance(result, list) and len(result) > 0:
            return OdooCustomer.from_odoo(result[0])

        return None

    async def lookup_customer_by_serial(self, serial: str) -> OdooCustomer | None:
        """Look up customer by device serial number.

        Searches stock.lot for the serial, then traces back to the customer
        through delivery orders.

        Args:
            serial: Device serial number.

        Returns:
            OdooCustomer or None if not found.
        """
        # First, find the lot (serial) in stock.lot
        lot_result = await self.execute_method(
            model="stock.lot",
            method="search_read",
            kwargs={
                "domain": [["name", "=", serial]],
                "fields": ["id", "name", "product_id"],
                "limit": 1,
            },
            use_cache=True,
        )

        if not lot_result or not isinstance(lot_result, list) or len(lot_result) == 0:
            return None

        lot_id = lot_result[0]["id"]

        # Find delivery picking that includes this serial
        # stock.move.line tracks lot assignments in pickings
        move_result = await self.execute_method(
            model="stock.move.line",
            method="search_read",
            kwargs={
                "domain": [
                    ["lot_id", "=", lot_id],
                    ["state", "=", "done"],
                ],
                "fields": ["picking_id"],
                "limit": 1,
                "order": "date desc",
            },
            use_cache=True,
        )

        if not move_result or not isinstance(move_result, list) or len(move_result) == 0:
            return None

        picking_id = move_result[0]["picking_id"]
        if isinstance(picking_id, list):
            picking_id = picking_id[0]

        if not picking_id:
            return None

        # Get partner from the picking
        picking_result = await self.execute_method(
            model="stock.picking",
            method="read",
            args=[[picking_id]],
            kwargs={"fields": ["partner_id"]},
            use_cache=True,
        )

        if not picking_result or not isinstance(picking_result, list) or len(picking_result) == 0:
            return None

        partner_id = picking_result[0].get("partner_id")
        if isinstance(partner_id, list):
            partner_id = partner_id[0]

        if not partner_id:
            return None

        # Get full partner info
        partner_result = await self.execute_method(
            model="res.partner",
            method="read",
            args=[[partner_id]],
            kwargs={"fields": self.PARTNER_FIELDS},
            use_cache=True,
        )

        if partner_result and isinstance(partner_result, list) and len(partner_result) > 0:
            return OdooCustomer.from_odoo(partner_result[0])

        return None

    async def create_customer(
        self,
        name: str,
        email: str,
        company: str | None = None,
        country_id: int | None = None,
        phone: str | None = None,
    ) -> OdooCustomer:
        """Create a new customer in Odoo.

        Args:
            name: Customer name.
            email: Email address.
            company: Company name (creates company + contact if provided).
            country_id: Odoo country ID.
            phone: Phone number.

        Returns:
            Created OdooCustomer.
        """
        values: dict[str, Any] = {
            "name": name,
            "email": email,
        }

        if phone:
            values["phone"] = phone

        if country_id:
            values["country_id"] = country_id

        if company:
            # First create or find the company
            existing_company = await self.execute_method(
                model="res.partner",
                method="search_read",
                kwargs={
                    "domain": [
                        ["name", "=ilike", company],
                        ["is_company", "=", True],
                    ],
                    "fields": ["id"],
                    "limit": 1,
                },
            )

            has_existing = (
                existing_company and isinstance(existing_company, list)
                and len(existing_company) > 0
            )
            if has_existing:
                parent_id = existing_company[0]["id"]
            else:
                # Create the company
                company_values = {
                    "name": company,
                    "is_company": True,
                }
                if country_id:
                    company_values["country_id"] = country_id

                parent_id = await self.execute_method(
                    model="res.partner",
                    method="create",
                    args=[[company_values]],
                )
                if isinstance(parent_id, list):
                    parent_id = parent_id[0]

            values["parent_id"] = parent_id
            values["is_company"] = False
        else:
            values["is_company"] = False

        # Create the contact
        customer_id = await self.execute_method(
            model="res.partner",
            method="create",
            args=[[values]],
        )

        if isinstance(customer_id, list):
            customer_id = customer_id[0]

        # Fetch the created customer
        result = await self.execute_method(
            model="res.partner",
            method="read",
            args=[[customer_id]],
            kwargs={"fields": self.PARTNER_FIELDS},
        )

        return OdooCustomer.from_odoo(result[0])

    async def get_customer_contacts(self, partner_id: int) -> list[OdooContact]:
        """Get contacts for a company partner.

        Args:
            partner_id: Company partner ID.

        Returns:
            List of contacts.
        """
        result = await self.execute_method(
            model="res.partner",
            method="search_read",
            kwargs={
                "domain": [
                    ["parent_id", "=", partner_id],
                    ["is_company", "=", False],
                ],
                "fields": ["id", "name", "email", "phone", "mobile", "function"],
                "limit": 50,
            },
            use_cache=True,
        )

        if result and isinstance(result, list):
            return [OdooContact.from_odoo(r) for r in result]

        return []

    # =========================================================================
    # Product/Sales Operations
    # =========================================================================

    async def search_products(
        self,
        query: str,
        category: str | None = None,
        limit: int = 20,
    ) -> list[OdooProduct]:
        """Search for products.

        Args:
            query: Search query (matches name or default_code).
            category: Optional category filter.
            limit: Maximum results.

        Returns:
            List of matching products.
        """
        domain: list[Any] = [
            "|",
            ["name", "ilike", query],
            ["default_code", "ilike", query],
        ]

        if category:
            domain.insert(0, ["categ_id.name", "ilike", category])
            domain.insert(0, "&")

        result = await self.execute_method(
            model="product.product",
            method="search_read",
            kwargs={
                "domain": domain,
                "fields": self.PRODUCT_FIELDS,
                "limit": limit,
            },
            use_cache=True,
        )

        if result and isinstance(result, list):
            return [OdooProduct.from_odoo(r) for r in result]

        return []

    async def get_customer_purchases(
        self,
        partner_id: int,
        limit: int = 20,
    ) -> list[OdooPurchase]:
        """Get purchase history for a customer.

        Optimized to batch-fetch all order lines in a single request instead of
        N+1 queries (one per order).

        Args:
            partner_id: Customer partner ID.
            limit: Maximum results.

        Returns:
            List of purchases (sale orders in 'sale' or 'done' state).
        """
        result = await self.execute_method(
            model="sale.order",
            method="search_read",
            kwargs={
                "domain": [
                    ["partner_id", "=", partner_id],
                    ["state", "in", ["sale", "done"]],
                ],
                "fields": ["id", "name", "date_order", "state", "amount_total", "order_line"],
                "limit": limit,
                "order": "date_order desc",
            },
            use_cache=True,
        )

        if not result or not isinstance(result, list):
            return []

        # Collect all order line IDs across all orders for batch fetch
        all_line_ids: list[int] = []
        order_to_lines: dict[int, list[int]] = {}

        for order in result:
            order_id = order.get("id", 0)
            order_lines = order.get("order_line", [])
            if order_lines and isinstance(order_lines, list):
                order_to_lines[order_id] = order_lines
                all_line_ids.extend(order_lines)

        # Batch fetch all order lines in a single request (was N+1, now 1)
        line_to_product: dict[int, str] = {}
        if all_line_ids:
            lines_result = await self.execute_method(
                model="sale.order.line",
                method="read",
                args=[all_line_ids],
                kwargs={"fields": ["id", "product_id"]},
                use_cache=True,
            )
            if lines_result and isinstance(lines_result, list):
                for line in lines_result:
                    line_id = line.get("id")
                    product = line.get("product_id")
                    if line_id and isinstance(product, list) and len(product) > 1:
                        line_to_product[line_id] = product[1]

        # Build purchases with product names from batch-fetched data
        purchases = []
        for order in result:
            order_id = order.get("id", 0)
            order_line_ids = order_to_lines.get(order_id, [])
            product_names = [
                line_to_product[lid] for lid in order_line_ids
                if lid in line_to_product
            ]

            order_date_raw = order.get("date_order")
            purchases.append(OdooPurchase(
                id=order_id,
                name=order.get("name", ""),
                date=(
                    datetime.fromisoformat(order_date_raw)
                    if order_date_raw else None
                ),
                state=order.get("state", ""),
                amount_total=order.get("amount_total", 0.0),
                product_names=product_names,
            ))

        return purchases

    async def create_quotation(
        self,
        partner_id: int,
        lines: list[OdooQuotationLine],
        validity_days: int = 30,
        note: str | None = None,
    ) -> OdooQuotation:
        """Create a sales quotation.

        Args:
            partner_id: Customer partner ID.
            lines: List of quotation lines.
            validity_days: Quotation validity in days.
            note: Optional note on the quotation.

        Returns:
            Created quotation.
        """
        from datetime import timedelta

        # Prepare order lines in Odoo format
        order_lines: list[tuple[int, int, dict[str, Any]]] = []
        for line in lines:
            line_values: dict[str, Any] = {
                "product_id": line.product_id,
                "product_uom_qty": line.quantity,
            }
            if line.name:
                line_values["name"] = line.name
            if line.price_unit is not None:
                line_values["price_unit"] = line.price_unit
            if line.discount > 0:
                line_values["discount"] = line.discount

            # (0, 0, {...}) format for creating new records in one2many
            order_lines.append((0, 0, line_values))

        values: dict[str, Any] = {
            "partner_id": partner_id,
            "order_line": order_lines,
        }

        if validity_days > 0:
            validity_date_str = (
                datetime.now() + timedelta(days=validity_days)
            ).date().isoformat()
            values["validity_date"] = validity_date_str

        if note:
            values["note"] = note

        # Create the quotation
        order_id = await self.execute_method(
            model="sale.order",
            method="create",
            args=[[values]],
        )

        if isinstance(order_id, list):
            order_id = order_id[0]

        # Fetch the created quotation
        return await self.get_quotation(order_id)

    async def get_quotation(self, quotation_id: int) -> OdooQuotation:
        """Get a quotation by ID.

        Args:
            quotation_id: Sale order ID.

        Returns:
            Quotation details.
        """
        result = await self.execute_method(
            model="sale.order",
            method="read",
            args=[[quotation_id]],
            kwargs={"fields": self.ORDER_FIELDS},
        )

        if not result or not isinstance(result, list) or len(result) == 0:
            raise OdooMCPAPIError(f"Quotation {quotation_id} not found")

        return OdooQuotation.from_odoo(result[0])

    # =========================================================================
    # Support/Repair Operations
    # =========================================================================

    async def check_warranty_status(self, serial: str) -> WarrantyStatus:
        """Check warranty status for a serial number.

        Optimized to trace serial → move → picking → partner in a single flow,
        avoiding the redundant lookup_customer_by_serial() call.

        Args:
            serial: Device serial number.

        Returns:
            Warranty status information.
        """
        from datetime import timedelta

        # Find the lot (serial)
        lot_result = await self.execute_method(
            model="stock.lot",
            method="search_read",
            kwargs={
                "domain": [["name", "=", serial]],
                "fields": ["id", "name", "product_id"],
                "limit": 1,
            },
            use_cache=True,
        )

        status = WarrantyStatus(serial=serial)

        if not lot_result or not isinstance(lot_result, list) or len(lot_result) == 0:
            return status

        lot = lot_result[0]
        product = lot.get("product_id")

        if isinstance(product, list) and len(product) > 1:
            status.product_id = product[0]
            status.product_name = product[1]

        lot_id = lot["id"]

        # Find the delivery move with picking_id (needed for partner lookup)
        move_result = await self.execute_method(
            model="stock.move.line",
            method="search_read",
            kwargs={
                "domain": [
                    ["lot_id", "=", lot_id],
                    ["state", "=", "done"],
                ],
                "fields": ["move_id", "date", "picking_id"],  # Include picking_id
                "limit": 1,
                "order": "date asc",  # First delivery
            },
            use_cache=True,
        )

        if not move_result or not isinstance(move_result, list) or len(move_result) == 0:
            return status

        move_data = move_result[0]

        # Extract purchase date and calculate warranty
        if move_data.get("date"):
            purchase_date = datetime.fromisoformat(move_data["date"].replace("Z", "+00:00"))
            status.purchase_date = purchase_date

            # 2-year warranty
            warranty_end = purchase_date + timedelta(days=730)
            status.warranty_end = warranty_end
            status.is_under_warranty = warranty_end > datetime.now(warranty_end.tzinfo or None)

        # Get customer from picking → partner chain (avoids redundant serial lookup)
        picking_id = move_data.get("picking_id")
        if isinstance(picking_id, list):
            picking_id = picking_id[0]

        if picking_id:
            picking_result = await self.execute_method(
                model="stock.picking",
                method="read",
                args=[[picking_id]],
                kwargs={"fields": ["partner_id"]},
                use_cache=True,
            )

            if picking_result and isinstance(picking_result, list) and len(picking_result) > 0:
                partner_id = picking_result[0].get("partner_id")
                if isinstance(partner_id, list) and len(partner_id) > 1:
                    status.customer_id = partner_id[0]
                    status.customer_name = partner_id[1]

        return status

    async def get_existing_repairs(
        self,
        partner_id: int,
        limit: int = 20,
    ) -> list[OdooRepair]:
        """Get existing repair orders for a customer.

        Args:
            partner_id: Customer partner ID.
            limit: Maximum results.

        Returns:
            List of repair orders.
        """
        result = await self.execute_method(
            model="repair.order",
            method="search_read",
            kwargs={
                "domain": [["partner_id", "=", partner_id]],
                "fields": self.REPAIR_FIELDS,
                "limit": limit,
                "order": "create_date desc",
            },
            use_cache=True,
        )

        if result and isinstance(result, list):
            return [OdooRepair.from_odoo(r) for r in result]

        return []

    async def create_repair(
        self,
        partner_id: int,
        product_id: int,
        description: str,
        lot_id: int | None = None,
    ) -> OdooRepair:
        """Create a repair order.

        Args:
            partner_id: Customer partner ID.
            product_id: Product ID to repair.
            description: Problem description.
            lot_id: Optional serial number (stock.lot ID).

        Returns:
            Created repair order.
        """
        values: dict[str, Any] = {
            "partner_id": partner_id,
            "product_id": product_id,
            "description": description,
        }

        if lot_id:
            values["lot_id"] = lot_id

        repair_id = await self.execute_method(
            model="repair.order",
            method="create",
            args=[[values]],
        )

        if isinstance(repair_id, list):
            repair_id = repair_id[0]

        # Fetch the created repair
        result = await self.execute_method(
            model="repair.order",
            method="read",
            args=[[repair_id]],
            kwargs={"fields": self.REPAIR_FIELDS},
        )

        return OdooRepair.from_odoo(result[0])

    # =========================================================================
    # Serial Number Operations
    # =========================================================================

    SERIAL_FIELDS = [
        "id", "name", "product_id", "company_id", "create_date"
    ]

    async def search_serials(
        self,
        query: str | None = None,
        product_id: int | None = None,
        customer_id: int | None = None,
        limit: int = 20,
    ) -> list[OdooSerial]:
        """Search for serial numbers in Odoo.

        Optimized with parallel enrichment using asyncio.gather() instead of
        sequential awaits.

        Args:
            query: Partial serial number search (uses ilike).
            product_id: Filter by product ID.
            customer_id: Filter by customer (via delivery history).
            limit: Maximum results (1-100, default 20).

        Returns:
            List of matching serial numbers with delivery info.
        """
        import asyncio

        limit = max(1, min(100, limit))

        # If searching by customer, we need to find lots via delivery
        if customer_id:
            return await self._search_serials_by_customer(customer_id, query, limit)

        # Build domain for direct lot search
        domain: list[Any] = []
        if query:
            domain.append(["name", "ilike", query])
        if product_id:
            domain.append(["product_id", "=", product_id])

        result = await self.execute_method(
            model="stock.lot",
            method="search_read",
            kwargs={
                "domain": domain,
                "fields": self.SERIAL_FIELDS,
                "limit": limit,
                "order": "create_date desc",
            },
            use_cache=True,
        )

        if not result or not isinstance(result, list):
            return []

        # Create serial objects
        serials = [OdooSerial.from_odoo(lot_data) for lot_data in result]

        # Parallel enrichment (was sequential, now concurrent)
        await asyncio.gather(
            *[self._enrich_serial_delivery_info(serial) for serial in serials]
        )

        return serials

    async def _search_serials_by_customer(
        self,
        customer_id: int,
        query: str | None = None,
        limit: int = 20,
    ) -> list[OdooSerial]:
        """Search serials by customer delivery history.

        Args:
            customer_id: Customer partner ID.
            query: Optional serial number filter.
            limit: Maximum results.

        Returns:
            List of serials delivered to this customer.
        """
        # Find deliveries to this customer
        picking_result = await self.execute_method(
            model="stock.picking",
            method="search_read",
            kwargs={
                "domain": [
                    ["partner_id", "=", customer_id],
                    ["state", "=", "done"],
                    ["picking_type_code", "=", "outgoing"],
                ],
                "fields": ["id", "name", "date_done"],
                "limit": 50,
                "order": "date_done desc",
            },
            use_cache=True,
        )

        if not picking_result or not isinstance(picking_result, list):
            return []

        picking_ids = [p["id"] for p in picking_result]
        picking_map = {p["id"]: p for p in picking_result}

        # Find move lines with lots from these pickings
        move_domain: list[Any] = [
            ["picking_id", "in", picking_ids],
            ["lot_id", "!=", False],
            ["state", "=", "done"],
        ]

        move_result = await self.execute_method(
            model="stock.move.line",
            method="search_read",
            kwargs={
                "domain": move_domain,
                "fields": ["lot_id", "picking_id"],
                "limit": limit * 2,  # Over-fetch for filtering
            },
            use_cache=True,
        )

        if not move_result or not isinstance(move_result, list):
            return []

        # Get unique lot IDs
        lot_ids = list({
            m["lot_id"][0] if isinstance(m["lot_id"], list) else m["lot_id"]
            for m in move_result if m.get("lot_id")
        })

        if not lot_ids:
            return []

        # Fetch lot details
        lot_domain: list[Any] = [["id", "in", lot_ids]]
        if query:
            lot_domain.append(["name", "ilike", query])

        lot_result = await self.execute_method(
            model="stock.lot",
            method="search_read",
            kwargs={
                "domain": lot_domain,
                "fields": self.SERIAL_FIELDS,
                "limit": limit,
            },
            use_cache=True,
        )

        serials = []
        if lot_result and isinstance(lot_result, list):
            # Build lot_id to picking mapping
            lot_to_picking: dict[int, dict[str, Any]] = {}
            for move in move_result:
                lot_id = move["lot_id"]
                if isinstance(lot_id, list):
                    lot_id = lot_id[0]
                picking_id = move["picking_id"]
                if isinstance(picking_id, list):
                    picking_id = picking_id[0]
                if lot_id and picking_id:
                    lot_to_picking[lot_id] = picking_map.get(picking_id, {})

            for lot_data in lot_result:
                serial = OdooSerial.from_odoo(lot_data)
                # Add customer info
                serial.customer_id = customer_id
                # Get customer name
                customer_result = await self.execute_method(
                    model="res.partner",
                    method="read",
                    args=[[customer_id]],
                    kwargs={"fields": ["name"]},
                    use_cache=True,
                )
                if customer_result and isinstance(customer_result, list):
                    serial.customer_name = customer_result[0].get("name")

                # Add delivery info from mapping
                picking_info = lot_to_picking.get(serial.id)
                if picking_info:
                    serial.delivery_ref = picking_info.get("name")
                    date_done = picking_info.get("date_done")
                    if date_done:
                        serial.delivery_date = datetime.fromisoformat(
                            date_done.replace("Z", "+00:00")
                        )

                serials.append(serial)

        return serials

    async def _enrich_serial_delivery_info(self, serial: OdooSerial) -> None:
        """Enrich serial with delivery information.

        Args:
            serial: Serial to enrich (modified in place).
        """
        # Find most recent delivery
        move_result = await self.execute_method(
            model="stock.move.line",
            method="search_read",
            kwargs={
                "domain": [
                    ["lot_id", "=", serial.id],
                    ["state", "=", "done"],
                ],
                "fields": ["picking_id", "date"],
                "limit": 1,
                "order": "date desc",
            },
            use_cache=True,
        )

        if not move_result or not isinstance(move_result, list) or len(move_result) == 0:
            return

        picking_id = move_result[0].get("picking_id")
        if isinstance(picking_id, list):
            picking_id = picking_id[0]

        if not picking_id:
            return

        # Get picking details with partner
        picking_result = await self.execute_method(
            model="stock.picking",
            method="read",
            args=[[picking_id]],
            kwargs={"fields": ["partner_id", "name", "date_done"]},
            use_cache=True,
        )

        if not picking_result or not isinstance(picking_result, list):
            return

        picking = picking_result[0]
        serial.delivery_ref = picking.get("name")

        date_done = picking.get("date_done")
        if date_done:
            serial.delivery_date = datetime.fromisoformat(
                date_done.replace("Z", "+00:00")
            )

        partner = picking.get("partner_id")
        if isinstance(partner, list) and len(partner) > 1:
            serial.customer_id = partner[0]
            serial.customer_name = partner[1]

    async def get_serial_info(self, serial: str) -> OdooSerial | None:
        """Get detailed information about a serial number.

        Args:
            serial: The serial number string.

        Returns:
            OdooSerial with full details or None if not found.
        """
        result = await self.execute_method(
            model="stock.lot",
            method="search_read",
            kwargs={
                "domain": [["name", "=", serial]],
                "fields": self.SERIAL_FIELDS,
                "limit": 1,
            },
            use_cache=True,
        )

        if not result or not isinstance(result, list) or len(result) == 0:
            return None

        serial_obj = OdooSerial.from_odoo(result[0])
        await self._enrich_serial_delivery_info(serial_obj)

        return serial_obj

    async def get_product_serials(
        self,
        product_id: int,
        limit: int = 50,
    ) -> list[OdooSerial]:
        """Get all serial numbers for a product.

        Optimized with parallel enrichment using asyncio.gather().

        Args:
            product_id: Odoo product ID.
            limit: Maximum results (1-100, default 50).

        Returns:
            List of serial numbers for this product.
        """
        import asyncio

        limit = max(1, min(100, limit))

        result = await self.execute_method(
            model="stock.lot",
            method="search_read",
            kwargs={
                "domain": [["product_id", "=", product_id]],
                "fields": self.SERIAL_FIELDS,
                "limit": limit,
                "order": "create_date desc",
            },
            use_cache=True,
        )

        if not result or not isinstance(result, list):
            return []

        # Create serial objects
        serials = [OdooSerial.from_odoo(lot_data) for lot_data in result]

        # Parallel enrichment (was sequential, now concurrent)
        await asyncio.gather(
            *[self._enrich_serial_delivery_info(serial) for serial in serials]
        )

        return serials

    # =========================================================================
    # Cache Management
    # =========================================================================

    def clear_cache(self) -> None:
        """Clear the read operation cache."""
        self._cache.invalidate()
        logger.info("Odoo MCP client cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats()


# Module-level singleton
_client: OdooMCPClient | None = None


def get_odoo_mcp_client() -> OdooMCPClient:
    """Get or create the Odoo MCP client singleton.

    Returns:
        OdooMCPClient instance.

    Raises:
        RuntimeError: If Odoo MCP integration is not enabled.
    """
    global _client

    settings = get_settings()
    if not settings.odoo_mcp_enabled:
        raise RuntimeError("Odoo MCP integration is not enabled. Set ODOO_MCP_ENABLED=true")

    if _client is None:
        _client = OdooMCPClient()
        logger.info(
            "Odoo MCP client initialized",
            url=_client._base_url,
            cache_ttl=_client._cache_ttl,
        )

    return _client


async def close_odoo_mcp_client() -> None:
    """Close the Odoo MCP client singleton and release resources.

    Safe to call even if client was never initialized.
    """
    global _client

    if _client is not None:
        await _client.close()
        _client = None
        logger.info("Odoo MCP client closed")
