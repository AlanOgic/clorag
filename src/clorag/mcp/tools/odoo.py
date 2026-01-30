"""Odoo MCP tools for CLORAG MCP server.

These tools expose Odoo CRM/ERP functionality through the CLORAG MCP server,
enabling Claude Desktop to interact with Odoo for customer lookup, sales
operations, and support workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from clorag.config import get_settings

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from clorag.mcp.server import MCPServices

logger = structlog.get_logger(__name__)


def register_odoo_tools(mcp: FastMCP[MCPServices]) -> None:
    """Register Odoo-related MCP tools.

    Tools are only registered if odoo_mcp_enabled is True in settings.

    Args:
        mcp: FastMCP server instance to register tools on.
    """
    settings = get_settings()
    if not settings.odoo_mcp_enabled:
        logger.info("Odoo MCP tools not registered (ODOO_MCP_ENABLED=false)")
        return

    logger.info("Registering Odoo MCP tools")

    # =========================================================================
    # Customer Tools
    # =========================================================================

    @mcp.tool()
    async def lookup_customer(
        email: str | None = None,
        serial: str | None = None,
    ) -> dict[str, Any]:
        """Look up a customer in Odoo CRM.

        Find customer information by email address or device serial number.
        Serial lookup traces the device through stock/delivery records.

        Args:
            email: Customer email address to search.
            serial: Device serial number to trace back to customer.

        Returns:
            Customer information or not found message.
        """
        from clorag.services.odoo_mcp_client import (
            OdooMCPError,
            get_odoo_mcp_client,
        )

        if not email and not serial:
            return {"error": "Either email or serial must be provided"}

        try:
            client = get_odoo_mcp_client()

            if email:
                customer = await client.lookup_customer_by_email(email)
            else:
                customer = await client.lookup_customer_by_serial(serial)  # type: ignore[arg-type]

            if customer:
                return {
                    "found": True,
                    "customer": {
                        "id": customer.id,
                        "name": customer.name,
                        "email": customer.email,
                        "phone": customer.phone,
                        "mobile": customer.mobile,
                        "company": customer.company_name,
                        "country": customer.country,
                        "is_company": customer.is_company,
                        "vat": customer.vat,
                    },
                }
            else:
                search_by = f"email '{email}'" if email else f"serial '{serial}'"
                return {
                    "found": False,
                    "message": f"No customer found for {search_by}",
                }

        except OdooMCPError as e:
            logger.error("Odoo customer lookup failed", error=str(e))
            return {"error": str(e)}

    @mcp.tool()
    async def create_customer(
        name: str,
        email: str,
        company: str | None = None,
        country_id: int | None = None,
        phone: str | None = None,
    ) -> dict[str, Any]:
        """Create a new customer in Odoo CRM.

        Creates a contact record. If company is provided, will find or create
        the company and link the contact to it.

        Args:
            name: Customer/contact name.
            email: Email address (required).
            company: Company name (optional, will create/link if provided).
            country_id: Odoo country ID (e.g., 233 for USA, 75 for France).
            phone: Phone number.

        Returns:
            Created customer information.
        """
        from clorag.services.odoo_mcp_client import (
            OdooMCPError,
            get_odoo_mcp_client,
        )

        try:
            client = get_odoo_mcp_client()

            customer = await client.create_customer(
                name=name,
                email=email,
                company=company,
                country_id=country_id,
                phone=phone,
            )

            return {
                "success": True,
                "customer": {
                    "id": customer.id,
                    "name": customer.name,
                    "email": customer.email,
                    "company": customer.company_name,
                    "country": customer.country,
                },
            }

        except OdooMCPError as e:
            logger.error("Odoo customer creation failed", error=str(e))
            return {"error": str(e)}

    @mcp.tool()
    async def get_customer_contacts(customer_id: int) -> dict[str, Any]:
        """Get contacts for a company customer.

        Lists all contact persons linked to a company partner.

        Args:
            customer_id: Odoo partner ID of the company.

        Returns:
            List of contacts with name, email, phone, and job title.
        """
        from clorag.services.odoo_mcp_client import (
            OdooMCPError,
            get_odoo_mcp_client,
        )

        try:
            client = get_odoo_mcp_client()

            contacts = await client.get_customer_contacts(customer_id)

            return {
                "customer_id": customer_id,
                "total": len(contacts),
                "contacts": [
                    {
                        "id": c.id,
                        "name": c.name,
                        "email": c.email,
                        "phone": c.phone,
                        "mobile": c.mobile,
                        "function": c.function,
                    }
                    for c in contacts
                ],
            }

        except OdooMCPError as e:
            logger.error("Odoo contacts lookup failed", error=str(e))
            return {"error": str(e)}

    # =========================================================================
    # Sales Tools
    # =========================================================================

    @mcp.tool()
    async def search_products(
        query: str,
        category: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search for products in Odoo.

        Searches products by name or internal reference (SKU).

        Args:
            query: Search term (matches product name or default_code).
            category: Optional category name filter.
            limit: Maximum results (1-50, default 20).

        Returns:
            List of matching products with details.
        """
        from clorag.services.odoo_mcp_client import (
            OdooMCPError,
            get_odoo_mcp_client,
        )

        try:
            client = get_odoo_mcp_client()

            limit = max(1, min(50, limit))
            products = await client.search_products(
                query=query,
                category=category,
                limit=limit,
            )

            return {
                "query": query,
                "category": category,
                "total": len(products),
                "products": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "sku": p.default_code,
                        "price": p.list_price,
                        "category": p.categ_name,
                        "type": p.type,
                        "description": p.description,
                    }
                    for p in products
                ],
            }

        except OdooMCPError as e:
            logger.error("Odoo product search failed", error=str(e))
            return {"error": str(e)}

    @mcp.tool()
    async def get_purchase_history(
        customer_id: int,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Get purchase history for a customer.

        Lists confirmed sales orders for a customer, showing what products
        they have purchased.

        Args:
            customer_id: Odoo partner ID.
            limit: Maximum orders to return (1-50, default 20).

        Returns:
            List of purchases with order details and product names.
        """
        from clorag.services.odoo_mcp_client import (
            OdooMCPError,
            get_odoo_mcp_client,
        )

        try:
            client = get_odoo_mcp_client()

            limit = max(1, min(50, limit))
            purchases = await client.get_customer_purchases(
                partner_id=customer_id,
                limit=limit,
            )

            return {
                "customer_id": customer_id,
                "total": len(purchases),
                "purchases": [
                    {
                        "id": p.id,
                        "order_number": p.name,
                        "date": p.date.isoformat() if p.date else None,
                        "state": p.state,
                        "amount_total": p.amount_total,
                        "products": p.product_names,
                    }
                    for p in purchases
                ],
            }

        except OdooMCPError as e:
            logger.error("Odoo purchase history lookup failed", error=str(e))
            return {"error": str(e)}

    @mcp.tool()
    async def create_quotation(
        customer_id: int,
        products: list[dict[str, Any]],
        validity_days: int = 30,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Create a sales quotation in Odoo.

        Creates a draft quotation for a customer with the specified products.

        Args:
            customer_id: Odoo partner ID for the customer.
            products: List of products, each with:
                - product_id: Odoo product ID (required)
                - quantity: Quantity (default 1)
                - price: Override unit price (optional)
                - discount: Discount percentage (optional)
            validity_days: Quotation validity in days (default 30).
            note: Optional note to include on the quotation.

        Returns:
            Created quotation details with reference number.
        """
        from clorag.services.odoo_mcp_client import (
            OdooMCPError,
            OdooQuotationLine,
            get_odoo_mcp_client,
        )

        try:
            client = get_odoo_mcp_client()

            # Convert products to OdooQuotationLine objects
            lines = []
            for p in products:
                if "product_id" not in p:
                    return {"error": "Each product must have a product_id"}

                lines.append(OdooQuotationLine(
                    product_id=p["product_id"],
                    quantity=p.get("quantity", 1.0),
                    price_unit=p.get("price"),
                    discount=p.get("discount", 0.0),
                ))

            quotation = await client.create_quotation(
                partner_id=customer_id,
                lines=lines,
                validity_days=validity_days,
                note=note,
            )

            return {
                "success": True,
                "quotation": {
                    "id": quotation.id,
                    "reference": quotation.name,
                    "customer_id": quotation.partner_id,
                    "customer_name": quotation.partner_name,
                    "state": quotation.state.value,
                    "amount_total": quotation.amount_total,
                    "amount_untaxed": quotation.amount_untaxed,
                    "currency": quotation.currency,
                    "validity_date": (
                        quotation.validity_date.isoformat()
                        if quotation.validity_date else None
                    ),
                },
            }

        except OdooMCPError as e:
            logger.error("Odoo quotation creation failed", error=str(e))
            return {"error": str(e)}

    @mcp.tool()
    async def get_quotation(quotation_id: int) -> dict[str, Any]:
        """Get quotation details from Odoo.

        Args:
            quotation_id: Odoo sale order ID.

        Returns:
            Full quotation details including lines.
        """
        from clorag.services.odoo_mcp_client import (
            OdooMCPError,
            get_odoo_mcp_client,
        )

        try:
            client = get_odoo_mcp_client()

            quotation = await client.get_quotation(quotation_id)

            return {
                "quotation": {
                    "id": quotation.id,
                    "reference": quotation.name,
                    "customer_id": quotation.partner_id,
                    "customer_name": quotation.partner_name,
                    "state": quotation.state.value,
                    "date_order": (
                        quotation.date_order.isoformat()
                        if quotation.date_order else None
                    ),
                    "amount_total": quotation.amount_total,
                    "amount_untaxed": quotation.amount_untaxed,
                    "currency": quotation.currency,
                    "validity_date": (
                        quotation.validity_date.isoformat()
                        if quotation.validity_date else None
                    ),
                    "note": quotation.note,
                    "lines": quotation.lines,
                },
            }

        except OdooMCPError as e:
            logger.error("Odoo quotation lookup failed", error=str(e))
            return {"error": str(e)}

    # =========================================================================
    # Support Tools
    # =========================================================================

    @mcp.tool()
    async def check_warranty(serial: str) -> dict[str, Any]:
        """Check warranty status for a device serial number.

        Looks up the serial in Odoo stock, finds the original delivery date,
        and calculates warranty status.

        Args:
            serial: Device serial number.

        Returns:
            Warranty status including product, purchase date, and coverage.
        """
        from clorag.services.odoo_mcp_client import (
            OdooMCPError,
            get_odoo_mcp_client,
        )

        try:
            client = get_odoo_mcp_client()

            status = await client.check_warranty_status(serial)

            return {
                "serial": status.serial,
                "found": status.product_id is not None,
                "product_name": status.product_name,
                "product_id": status.product_id,
                "purchase_date": status.purchase_date.isoformat() if status.purchase_date else None,
                "warranty_end": status.warranty_end.isoformat() if status.warranty_end else None,
                "is_under_warranty": status.is_under_warranty,
                "customer_id": status.customer_id,
                "customer_name": status.customer_name,
            }

        except OdooMCPError as e:
            logger.error("Odoo warranty check failed", error=str(e))
            return {"error": str(e)}

    @mcp.tool()
    async def get_repair_history(
        customer_id: int,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Get repair order history for a customer.

        Lists existing repair orders for a customer, useful for understanding
        support history.

        Args:
            customer_id: Odoo partner ID.
            limit: Maximum repairs to return (1-50, default 20).

        Returns:
            List of repair orders with status and details.
        """
        from clorag.services.odoo_mcp_client import (
            OdooMCPError,
            get_odoo_mcp_client,
        )

        try:
            client = get_odoo_mcp_client()

            limit = max(1, min(50, limit))
            repairs = await client.get_existing_repairs(
                partner_id=customer_id,
                limit=limit,
            )

            return {
                "customer_id": customer_id,
                "total": len(repairs),
                "repairs": [
                    {
                        "id": r.id,
                        "reference": r.name,
                        "product": r.product_name,
                        "serial": r.lot_name,
                        "state": r.state,
                        "description": r.description,
                        "create_date": r.create_date.isoformat() if r.create_date else None,
                    }
                    for r in repairs
                ],
            }

        except OdooMCPError as e:
            logger.error("Odoo repair history lookup failed", error=str(e))
            return {"error": str(e)}

    @mcp.tool()
    async def create_repair(
        customer_id: int,
        product_id: int,
        description: str,
        lot_id: int | None = None,
    ) -> dict[str, Any]:
        """Create a repair order in Odoo.

        Creates a new repair order (RMA) for a customer's product.

        Args:
            customer_id: Odoo partner ID for the customer.
            product_id: Odoo product ID of the item to repair.
            description: Problem description for the repair.
            lot_id: Optional stock.lot ID for serial number tracking.

        Returns:
            Created repair order details.
        """
        from clorag.services.odoo_mcp_client import (
            OdooMCPError,
            get_odoo_mcp_client,
        )

        try:
            client = get_odoo_mcp_client()

            repair = await client.create_repair(
                partner_id=customer_id,
                product_id=product_id,
                description=description,
                lot_id=lot_id,
            )

            return {
                "success": True,
                "repair": {
                    "id": repair.id,
                    "reference": repair.name,
                    "customer_id": repair.partner_id,
                    "customer_name": repair.partner_name,
                    "product": repair.product_name,
                    "serial": repair.lot_name,
                    "state": repair.state,
                    "description": repair.description,
                },
            }

        except OdooMCPError as e:
            logger.error("Odoo repair creation failed", error=str(e))
            return {"error": str(e)}

    # =========================================================================
    # Serial Number Tools
    # =========================================================================

    @mcp.tool()
    async def search_serials(
        query: str | None = None,
        product_id: int | None = None,
        customer_id: int | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search for serial numbers in Odoo inventory.

        Search serial numbers (stock.lot records) by partial match, product,
        or customer. When searching by customer, finds all serials that have
        been delivered to that customer.

        Args:
            query: Partial serial number to search (uses wildcard matching).
            product_id: Filter serials by Odoo product ID.
            customer_id: Find serials delivered to this customer (partner ID).
            limit: Maximum results to return (1-100, default 20).

        Returns:
            List of serial numbers with product and delivery information.
        """
        from clorag.services.odoo_mcp_client import (
            OdooMCPError,
            get_odoo_mcp_client,
        )

        if not query and not product_id and not customer_id:
            return {
                "error": "At least one search parameter "
                "(query, product_id, or customer_id) is required"
            }

        try:
            client = get_odoo_mcp_client()

            serials = await client.search_serials(
                query=query,
                product_id=product_id,
                customer_id=customer_id,
                limit=limit,
            )

            return {
                "query": query,
                "product_id": product_id,
                "customer_id": customer_id,
                "total": len(serials),
                "serials": [
                    {
                        "id": s.id,
                        "serial": s.name,
                        "product_id": s.product_id,
                        "product_name": s.product_name,
                        "customer_id": s.customer_id,
                        "customer_name": s.customer_name,
                        "delivery_date": (
                            s.delivery_date.isoformat() if s.delivery_date else None
                        ),
                        "delivery_ref": s.delivery_ref,
                        "create_date": (
                            s.create_date.isoformat() if s.create_date else None
                        ),
                    }
                    for s in serials
                ],
            }

        except OdooMCPError as e:
            logger.error("Odoo serial search failed", error=str(e))
            return {"error": str(e)}

    @mcp.tool()
    async def get_serial_info(serial: str) -> dict[str, Any]:
        """Get detailed information about a specific serial number.

        Looks up a serial number in Odoo and retrieves full details including
        product information and delivery history (customer, date, reference).

        Args:
            serial: The exact serial number to look up.

        Returns:
            Full serial number details or not found message.
        """
        from clorag.services.odoo_mcp_client import (
            OdooMCPError,
            get_odoo_mcp_client,
        )

        try:
            client = get_odoo_mcp_client()

            result = await client.get_serial_info(serial)

            if result:
                return {
                    "found": True,
                    "serial": {
                        "id": result.id,
                        "serial": result.name,
                        "product_id": result.product_id,
                        "product_name": result.product_name,
                        "customer_id": result.customer_id,
                        "customer_name": result.customer_name,
                        "delivery_date": (
                            result.delivery_date.isoformat() if result.delivery_date else None
                        ),
                        "delivery_ref": result.delivery_ref,
                        "create_date": (
                            result.create_date.isoformat() if result.create_date else None
                        ),
                        "company_id": result.company_id,
                        "company_name": result.company_name,
                    },
                }
            else:
                return {
                    "found": False,
                    "message": f"Serial number '{serial}' not found in Odoo",
                }

        except OdooMCPError as e:
            logger.error("Odoo serial info lookup failed", error=str(e))
            return {"error": str(e)}

    @mcp.tool()
    async def get_product_serials(
        product_id: int,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get all serial numbers for a specific product.

        Lists all serial numbers (stock.lot records) associated with a product,
        with delivery information where available.

        Args:
            product_id: Odoo product ID to get serials for.
            limit: Maximum results to return (1-100, default 50).

        Returns:
            List of serial numbers for this product.
        """
        from clorag.services.odoo_mcp_client import (
            OdooMCPError,
            get_odoo_mcp_client,
        )

        try:
            client = get_odoo_mcp_client()

            serials = await client.get_product_serials(
                product_id=product_id,
                limit=limit,
            )

            return {
                "product_id": product_id,
                "total": len(serials),
                "serials": [
                    {
                        "id": s.id,
                        "serial": s.name,
                        "product_name": s.product_name,
                        "customer_id": s.customer_id,
                        "customer_name": s.customer_name,
                        "delivery_date": (
                            s.delivery_date.isoformat() if s.delivery_date else None
                        ),
                        "delivery_ref": s.delivery_ref,
                        "create_date": (
                            s.create_date.isoformat() if s.create_date else None
                        ),
                    }
                    for s in serials
                ],
            }

        except OdooMCPError as e:
            logger.error("Odoo product serials lookup failed", error=str(e))
            return {"error": str(e)}

    logger.info("Odoo MCP tools registered successfully")
