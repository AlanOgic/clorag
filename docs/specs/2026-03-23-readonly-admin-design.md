# Read-Only Admin Access

**Date**: 2026-03-23
**Status**: Approved

## Problem

The admin panel currently has a single access tier: full read+write via `ADMIN_PASSWORD`. There is no way to give someone view-only access to dashboards, analytics, camera data, settings, and other admin pages without also granting them the ability to modify data.

## Solution

Add a second password (`VIEWER_PASSWORD`) that grants read-only access to all admin pages. The same login form is used — the server determines the role based on which password matches.

## Auth Model

- New env var `VIEWER_PASSWORD` (+ Docker secret support via `VIEWER_PASSWORD_FILE`)
- Same login form, server checks both passwords with `secrets.compare_digest`
- Admin password checked first — if both are set to the same value, user gets full access
- Session token stores `"role": "admin"` or `"role": "viewer"`
- Brute force protection applies to both passwords (shared tracker per IP)

## Server-Side Enforcement

### New Dependency: `verify_admin_write()`

Extracts role from session token. Raises HTTP 403 ("Admin access required") if role is `"viewer"`. Applied to all mutating endpoints except `POST /login` and `POST /logout`.

### Endpoint Classification

**Admin-only (write) — 38 endpoints:**

| Router | Endpoints | Methods |
|--------|-----------|---------|
| cameras.py | import, create, update, delete, merge, duplicates-find | POST, PUT, DELETE |
| chunks.py | update, delete | PUT, DELETE |
| debug.py | reindex | POST |
| documents.py | upload, create, update, delete | POST, PUT, DELETE |
| drafts.py | generate, send, batch | POST |
| graph.py | delete relationship, update relationship | DELETE, PATCH |
| ingestion.py | run, cancel, delete-job | POST, DELETE |
| prompts.py | update, initialize, reload, rollback | POST, PUT |
| settings.py | update, initialize, reload, rollback, export | POST, PUT |
| support.py | delete | DELETE |
| terminology.py | scan, apply, update, rollback, delete | POST, PUT, DELETE |
| auth.py | backup | GET (admin-only despite being GET — it downloads raw databases) |

**Both roles (read) — all GET endpoints** (except backup):

All existing GET endpoints on admin routers remain accessible to both roles via the unchanged `verify_admin` dependency.

### Session Endpoint Change

`GET /api/admin/session` returns role information:

```json
{"authenticated": true, "role": "admin"}
{"authenticated": true, "role": "viewer"}
{"authenticated": false}
```

## Client-Side UX

### Template Context

All admin page routes in `pages.py` pass `is_viewer: bool` to template context, extracted from the session cookie.

### Disabled Buttons

Write-action buttons (edit, delete, merge, upload, run, apply, etc.) are rendered with:
- `disabled` HTML attribute when `is_viewer` is true
- `title="Admin access required"` tooltip
- CSS class `viewer-disabled` for consistent greyed-out styling

### JavaScript

- `AdminActions` stores the current role (fetched from session endpoint on page load)
- Before executing any mutation (`confirm-delete`, `submit-form`, `call` to write functions), checks role
- If viewer, shows a toast notification: "Admin access required" and blocks the action
- This is a UX convenience — server-side enforcement is the security boundary

### CSS

```css
.viewer-disabled {
    opacity: 0.5;
    cursor: not-allowed;
    pointer-events: none;
}
```

## Files Changed

| File | Change |
|------|--------|
| `config.py` | Add `viewer_password: SecretStr \| None` field, add to `_SECRET_FIELDS` |
| `auth/admin.py` | Add `verify_admin_write()` dependency, add role extraction helper |
| `auth/__init__.py` | Export `verify_admin_write` |
| `routers/admin/auth.py` | Check both passwords on login, include role in session token and session endpoint |
| `routers/admin/cameras.py` | `verify_admin` → `verify_admin_write` on 5 mutating endpoints |
| `routers/admin/chunks.py` | `verify_admin` → `verify_admin_write` on 2 mutating endpoints |
| `routers/admin/debug.py` | `verify_admin` → `verify_admin_write` on 1 mutating endpoint |
| `routers/admin/documents.py` | `verify_admin` → `verify_admin_write` on 4 mutating endpoints |
| `routers/admin/drafts.py` | `verify_admin` → `verify_admin_write` on 3 mutating endpoints |
| `routers/admin/graph.py` | `verify_admin` → `verify_admin_write` on 2 mutating endpoints |
| `routers/admin/ingestion.py` | `verify_admin` → `verify_admin_write` on 3 mutating endpoints |
| `routers/admin/prompts.py` | `verify_admin` → `verify_admin_write` on 5 mutating endpoints |
| `routers/admin/settings.py` | `verify_admin` → `verify_admin_write` on 5 mutating endpoints |
| `routers/admin/support.py` | `verify_admin` → `verify_admin_write` on 1 mutating endpoint |
| `routers/admin/terminology.py` | `verify_admin` → `verify_admin_write` on 5 mutating endpoints |
| `routers/pages.py` | Pass `is_viewer` to all admin template contexts |
| `static/js/admin.js` | Role-aware action gating, toast notification |
| `static/css/styles.css` | `.viewer-disabled` class |
| 34 admin templates | Conditional `disabled` + `viewer-disabled` class on write-action elements |

## Not Changed

- Public search/camera pages — unaffected
- CSRF system — still required for write endpoints (viewers never reach them)
- MCP server — separate auth via `MCP_API_KEY`
- OpenAI-compatible API — separate auth via `OPENAI_COMPAT_API_KEY`
- CLI agent — no admin auth involved

## Configuration

```bash
# .env
ADMIN_PASSWORD=full-access-password
VIEWER_PASSWORD=readonly-password   # Optional — if not set, viewer role unavailable
```

Docker secret: `VIEWER_PASSWORD_FILE` supported via existing `_SECRET_FIELDS` mechanism.
