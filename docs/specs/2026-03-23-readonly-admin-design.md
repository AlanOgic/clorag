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
- **Backward compatibility**: existing sessions without a `"role"` field are treated as `"admin"` to avoid locking out users with active pre-upgrade sessions
- Brute force protection applies to both passwords (shared tracker per IP)
- `VIEWER_PASSWORD` is never used for session signing — signing remains `admin_password`-derived
- Login logs include role for audit trail: `"Successful admin login"` vs `"Successful viewer login"`

## Server-Side Enforcement

### New Dependency: `verify_admin_write()`

Extracts role from session token. Raises HTTP 403 ("Admin access required") if role is `"viewer"`. Applied to all mutating endpoints except `POST /login` and `POST /logout`.

**Dependency ordering**: `verify_admin_write` should be evaluated BEFORE `verify_csrf` in endpoint signatures. Failing fast at 403 (role check) gives a clearer error than 403 (CSRF validation).

### Endpoint Classification

**Admin-only (write) — 36 endpoints:**

| Router | Endpoints | Methods |
|--------|-----------|---------|
| cameras.py | import, create, update, delete, merge | POST, PUT, DELETE |
| chunks.py | update, delete | PUT, DELETE |
| documents.py | upload, create, update, delete | POST, PUT, DELETE |
| drafts.py | generate, send, batch | POST |
| graph.py | delete relationship, update relationship | DELETE, PATCH |
| ingestion.py | run, cancel, delete-job | POST, DELETE |
| prompts.py | update, from-default, rollback, initialize, reload | PUT, POST |
| settings.py | update, update_metadata, initialize, reload, rollback | PUT, POST |
| support.py | delete | DELETE |
| terminology.py | scan, apply, update, rollback, delete | POST, PUT, DELETE |
| auth.py | backup | GET (admin-only despite being GET — downloads raw databases) |

**Both roles (read) — all GET endpoints** (except backup), plus:

| Router | Endpoint | Reason |
|--------|----------|--------|
| debug.py | `POST /search-debug` | Read-only search with JSON body, no mutation |
| prompts.py | `POST /prompts/test` | Preview-only variable substitution, no mutation |
| cameras.py | `GET /cameras/duplicates` | Read-only duplicate detection |

### Session Endpoint Change

`GET /api/admin/session` returns role information:

```json
{"authenticated": true, "role": "admin"}
{"authenticated": true, "role": "viewer"}
{"authenticated": false}
```

## Client-Side UX

### Template Context

A shared helper `get_admin_role(admin_session)` extracts the role from the session cookie. A Jinja2 context processor (or middleware) injects `is_viewer: bool` into all admin template contexts automatically, avoiding repetitive code across ~18 page handlers.

### Disabled Buttons

Write-action buttons (edit, delete, merge, upload, run, apply, etc.) are rendered with:
- `disabled` HTML attribute when `is_viewer` is true
- `title="Admin access required"` tooltip
- CSS class `viewer-disabled` for consistent greyed-out styling
- For `<a>` links styled as buttons, the `disabled` attribute alone is insufficient — `tabindex="-1"` and `aria-disabled="true"` are added

### Viewer-Only Pages

- `/admin/cameras/new` — redirect to cameras list (no point for viewers)
- `/admin/cameras/{id}/edit` — render in read-only mode (disabled form fields)

### Role Badge

A "Viewer" badge is shown in the admin navbar next to the user icon when the session role is `"viewer"`, so users immediately understand their access level.

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

Note: `pointer-events: none` blocks mouse clicks but not keyboard activation. The `disabled` HTML attribute handles `<button>` elements; the JS-level role check in `AdminActions` catches `data-action` elements activated via keyboard.

## Files Changed

| File | Change |
|------|--------|
| `config.py` | Add `viewer_password: SecretStr \| None` field, add to `_SECRET_FIELDS` |
| `auth/admin.py` | Add `verify_admin_write()` dependency, `get_admin_role()` helper |
| `auth/__init__.py` | Export `verify_admin_write`, `get_admin_role` in `__all__` |
| `routers/admin/__init__.py` | Update docstring to mention `verify_admin_write` for mutating endpoints |
| `routers/admin/auth.py` | Check both passwords on login, include role in session token and session endpoint |
| `routers/admin/cameras.py` | `verify_admin` → `verify_admin_write` on 5 mutating endpoints |
| `routers/admin/chunks.py` | `verify_admin` → `verify_admin_write` on 2 mutating endpoints |
| `routers/admin/documents.py` | `verify_admin` → `verify_admin_write` on 4 mutating endpoints |
| `routers/admin/drafts.py` | `verify_admin` → `verify_admin_write` on 3 mutating endpoints |
| `routers/admin/graph.py` | `verify_admin` → `verify_admin_write` on 2 mutating endpoints |
| `routers/admin/ingestion.py` | `verify_admin` → `verify_admin_write` on 3 mutating endpoints |
| `routers/admin/prompts.py` | `verify_admin` → `verify_admin_write` on 5 mutating endpoints (not test) |
| `routers/admin/settings.py` | `verify_admin` → `verify_admin_write` on 5 mutating endpoints |
| `routers/admin/support.py` | `verify_admin` → `verify_admin_write` on 1 mutating endpoint |
| `routers/admin/terminology.py` | `verify_admin` → `verify_admin_write` on 5 mutating endpoints |
| `routers/pages.py` | Inject `is_viewer` via context processor/middleware |
| `static/js/admin.js` | Role-aware action gating, toast notification, role badge |
| `static/css/styles.css` | `.viewer-disabled` class, role badge styling |
| Admin templates (`admin_*.html`) | Conditional `disabled` + `viewer-disabled` on write-action elements |

## Not Changed

- `routers/admin/debug.py` — `POST /search-debug` is read-only, keeps `verify_admin`
- Public search/camera pages — unaffected
- CSRF system — still required for write endpoints (viewers never reach them)
- MCP server — separate auth via `MCP_API_KEY`
- OpenAI-compatible API — separate auth via `OPENAI_COMPAT_API_KEY`
- CLI agent — no admin auth involved
- Session signing — remains derived from `admin_password`, not `viewer_password`

## Configuration

```bash
# .env
ADMIN_PASSWORD=full-access-password
VIEWER_PASSWORD=readonly-password   # Optional — if not set, viewer role unavailable
```

Docker secret: `VIEWER_PASSWORD_FILE` supported via existing `_SECRET_FIELDS` mechanism.
