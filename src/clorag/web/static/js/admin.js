/**
 * Admin utilities for CLORAG
 * Provides CSRF protection, theme toggle, and common admin functionality
 */

// Theme toggle - runs immediately to prevent flash of wrong theme
const ThemeToggle = {
    STORAGE_KEY: 'clorag-theme',

    init() {
        const saved = localStorage.getItem(this.STORAGE_KEY);
        if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
        }
    },

    toggle() {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        const next = isDark ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem(this.STORAGE_KEY, next);
        // Update button icon
        const btn = document.querySelector('.btn-theme-toggle');
        if (btn) btn.textContent = next === 'dark' ? '\u2600\uFE0F' : '\uD83C\uDF19';
    },

    isDark() {
        return document.documentElement.getAttribute('data-theme') === 'dark';
    },

    injectButton() {
        const nav = document.querySelector('.navbar__links');
        if (!nav) return;
        const btn = document.createElement('button');
        btn.className = 'btn-theme-toggle';
        btn.title = 'Toggle dark mode';
        btn.setAttribute('data-action', 'call');
        btn.setAttribute('data-fn', 'toggleTheme');
        btn.textContent = this.isDark() ? '\u2600\uFE0F' : '\uD83C\uDF19';
        nav.insertBefore(btn, nav.firstChild);
    }
};

// Apply theme immediately to prevent flash
ThemeToggle.init();

// Global function for data-action="call" dispatch
function toggleTheme() {
    ThemeToggle.toggle();
}

// CSRF token management
const AdminUtils = {
    csrfToken: null,
    csrfTokenExpiry: null,
    CSRF_REFRESH_INTERVAL: 45 * 60 * 1000, // Refresh every 45 minutes (token valid for 1 hour)

    /**
     * Initialize CSRF protection - call this on page load
     */
    async init() {
        await this.fetchCsrfToken();
        // Refresh token periodically
        setInterval(() => this.fetchCsrfToken(), this.CSRF_REFRESH_INTERVAL);
    },

    /**
     * Fetch a fresh CSRF token from the server
     */
    async fetchCsrfToken() {
        try {
            const res = await fetch('/api/admin/csrf-token');
            if (res.ok) {
                const data = await res.json();
                this.csrfToken = data.csrf_token;
                this.csrfTokenExpiry = Date.now() + (55 * 60 * 1000); // Mark as expiring in 55 minutes
                console.debug('CSRF token refreshed');
            }
        } catch (e) {
            console.error('Failed to fetch CSRF token:', e);
        }
    },

    /**
     * Get current CSRF token, refreshing if needed
     */
    async getToken() {
        if (!this.csrfToken || Date.now() > this.csrfTokenExpiry) {
            await this.fetchCsrfToken();
        }
        return this.csrfToken;
    },

    /**
     * Make a fetch request with CSRF protection
     * @param {string} url - The URL to fetch
     * @param {object} options - Fetch options
     * @returns {Promise<Response>}
     */
    async fetch(url, options = {}) {
        const headers = options.headers || {};

        // Add CSRF token for state-changing methods
        if (['POST', 'PUT', 'DELETE', 'PATCH'].includes((options.method || 'GET').toUpperCase())) {
            const token = await this.getToken();
            if (token) {
                headers['X-CSRF-Token'] = token;
            }
        }

        return fetch(url, {
            ...options,
            headers: headers
        });
    },

    /**
     * Make a JSON POST request with CSRF protection
     * @param {string} url - The URL to fetch
     * @param {object} data - JSON data to send
     * @returns {Promise<Response>}
     */
    async postJson(url, data) {
        return this.fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
    },

    /**
     * Make a JSON PUT request with CSRF protection
     * @param {string} url - The URL to fetch
     * @param {object} data - JSON data to send
     * @returns {Promise<Response>}
     */
    async putJson(url, data) {
        return this.fetch(url, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
    },

    /**
     * Make a DELETE request with CSRF protection
     * @param {string} url - The URL to delete
     * @returns {Promise<Response>}
     */
    async delete(url) {
        return this.fetch(url, {
            method: 'DELETE'
        });
    }
};

/**
 * Global Event Delegation System
 *
 * Use data-action attributes instead of inline onclick handlers to comply with CSP.
 *
 * Supported actions:
 * - data-action="toggle-section" data-target="sectionId" - Toggle collapsible section
 * - data-action="toggle-expand" data-target="elementId" - Toggle element visibility
 * - data-action="close-modal" data-modal="modalId" - Close a modal
 * - data-action="open-modal" data-modal="modalId" - Open a modal
 * - data-action="call" data-fn="functionName" data-args="arg1,arg2" - Call a global function
 * - data-action="confirm-delete" data-url="/api/..." data-redirect="/redirect" - Delete with confirmation
 * - data-action="submit-form" data-form="formId" - Submit a form
 * - data-action="add-row" data-container="containerId" data-name="fieldName" data-placeholder="..." - Add form row
 * - data-action="remove-row" - Remove parent row (for dynamic form fields)
 */
const AdminActions = {
    // Registry for page-specific action handlers
    handlers: {},

    /**
     * Register a custom action handler for the current page
     * @param {string} action - Action name
     * @param {function} handler - Handler function(element, event)
     */
    register(action, handler) {
        this.handlers[action] = handler;
    },

    /**
     * Toggle a collapsible section
     */
    toggleSection(target) {
        const content = document.getElementById(target + 'Content');
        const icon = document.getElementById(target + 'Icon');
        const label = document.getElementById(target + 'Label');

        if (content) {
            const isExpanded = content.classList.toggle('expanded');
            if (icon) icon.classList.toggle('rotated');
            if (label) label.textContent = isExpanded ? 'Hide' : 'Show';
        }
    },

    /**
     * Toggle element expanded state
     */
    toggleExpand(target) {
        const element = document.getElementById(target);
        const icon = document.getElementById('expandIcon' + target.replace(/\D/g, ''));

        if (element) {
            element.classList.toggle('expanded');
            if (icon) icon.classList.toggle('rotated');
        }
    },

    /**
     * Close a modal (with animation if available)
     */
    closeModal(modalId) {
        const modal = document.getElementById(modalId);
        if (!modal) return;
        if (typeof AdminAnimations !== 'undefined') {
            AdminAnimations.closeModalAnimated(modal);
        } else {
            modal.classList.add('hidden');
        }
    },

    /**
     * Open a modal
     */
    openModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.remove('hidden');
        }
    },

    /**
     * Call a global function by name.
     * Args may be provided either via ``data-args`` (comma-separated simple
     * values) or ``data-args-json`` (a JSON-encoded array of values). The
     * JSON form is preferred when any argument may contain commas, quotes,
     * or non-string values.
     */
    callFunction(fnName, args, argsJson) {
        const fn = window[fnName];
        if (typeof fn !== 'function') {
            console.warn(`AdminActions: Function "${fnName}" not found`);
            return;
        }
        let parsedArgs = [];
        if (argsJson) {
            try {
                parsedArgs = JSON.parse(argsJson);
                if (!Array.isArray(parsedArgs)) parsedArgs = [parsedArgs];
            } catch (e) {
                console.warn(`AdminActions: invalid data-args-json for "${fnName}":`, e);
                return;
            }
        } else if (args) {
            parsedArgs = args.split(',').map(a => {
                const trimmed = a.trim();
                if (/^\d+$/.test(trimmed)) return parseInt(trimmed, 10);
                if (/^\d+\.\d+$/.test(trimmed)) return parseFloat(trimmed);
                if (/^['"].*['"]$/.test(trimmed)) return trimmed.slice(1, -1);
                if (trimmed === 'null') return null;
                if (trimmed === 'true') return true;
                if (trimmed === 'false') return false;
                return trimmed;
            });
        }
        fn(...parsedArgs);
    },

    /**
     * Call a global function, passing the element as the first arg.
     * Use for cases that previously relied on `this` inside inline handlers.
     */
    callFunctionWithEl(element, fnName, args, argsJson) {
        const fn = window[fnName];
        if (typeof fn !== 'function') {
            console.warn(`AdminActions: Function "${fnName}" not found`);
            return;
        }
        let parsedArgs = [];
        if (argsJson) {
            try {
                parsedArgs = JSON.parse(argsJson);
                if (!Array.isArray(parsedArgs)) parsedArgs = [parsedArgs];
            } catch (e) {
                console.warn(`AdminActions: invalid data-args-json for "${fnName}":`, e);
                return;
            }
        } else if (args) {
            parsedArgs = args.split(',').map(a => {
                const trimmed = a.trim();
                if (/^\d+$/.test(trimmed)) return parseInt(trimmed, 10);
                if (/^\d+\.\d+$/.test(trimmed)) return parseFloat(trimmed);
                if (/^['"].*['"]$/.test(trimmed)) return trimmed.slice(1, -1);
                if (trimmed === 'null') return null;
                if (trimmed === 'true') return true;
                if (trimmed === 'false') return false;
                return trimmed;
            });
        }
        fn(element, ...parsedArgs);
    },

    /**
     * Delete with confirmation
     */
    async confirmDelete(url, redirect, message) {
        const confirmMsg = message || 'Are you sure you want to delete this item?';
        if (!confirm(confirmMsg)) return;

        try {
            const res = await AdminUtils.delete(url);
            if (res.ok) {
                if (redirect) {
                    window.location.href = redirect;
                } else {
                    window.location.reload();
                }
            } else {
                const data = await res.json().catch(() => ({}));
                alert(data.detail || 'Delete failed');
            }
        } catch (e) {
            console.error('Delete failed:', e);
            alert('Delete failed: ' + e.message);
        }
    },

    /**
     * Add a dynamic row to a container
     */
    addRow(containerId, fieldName, placeholder) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const row = document.createElement('div');
        row.className = 'form-row';
        row.innerHTML = `
            <input type="text" name="${fieldName}" placeholder="${placeholder || ''}">
            <button type="button" class="btn-remove" data-action="remove-row">X</button>
        `;
        container.appendChild(row);
    },

    /**
     * Remove the parent row of an element
     */
    removeRow(element) {
        const row = element.closest('.form-row');
        if (row) row.remove();
    },

    /**
     * Handle click events via delegation
     */
    handleClick(event) {
        // Find the closest element with data-action
        const actionEl = event.target.closest('[data-action]');
        if (!actionEl) return;

        const action = actionEl.dataset.action;

        // Check for page-specific handler first
        if (this.handlers[action]) {
            event.preventDefault();
            this.handlers[action](actionEl, event);
            return;
        }

        // Built-in actions
        switch (action) {
            case 'toggle-section':
                event.preventDefault();
                this.toggleSection(actionEl.dataset.target || actionEl.dataset.section);
                break;

            case 'toggle-expand':
                event.preventDefault();
                this.toggleExpand(actionEl.dataset.target);
                break;

            case 'close-modal':
                event.preventDefault();
                this.closeModal(actionEl.dataset.modal);
                break;

            case 'open-modal':
                event.preventDefault();
                this.openModal(actionEl.dataset.modal);
                break;

            case 'call':
                event.preventDefault();
                this.callFunction(
                    actionEl.dataset.fn,
                    actionEl.dataset.args,
                    actionEl.dataset.argsJson,
                );
                break;

            case 'call-with-el':
                event.preventDefault();
                this.callFunctionWithEl(
                    actionEl,
                    actionEl.dataset.fn,
                    actionEl.dataset.args,
                    actionEl.dataset.argsJson,
                );
                break;

            case 'stop-propagation':
                event.stopPropagation();
                break;

            case 'confirm-delete':
                event.preventDefault();
                this.confirmDelete(
                    actionEl.dataset.url,
                    actionEl.dataset.redirect,
                    actionEl.dataset.message
                );
                break;

            case 'add-row':
                event.preventDefault();
                this.addRow(
                    actionEl.dataset.container,
                    actionEl.dataset.name,
                    actionEl.dataset.placeholder
                );
                break;

            case 'remove-row':
                event.preventDefault();
                this.removeRow(actionEl);
                break;

            case 'submit-form':
                event.preventDefault();
                const form = document.getElementById(actionEl.dataset.form);
                if (form) form.submit();
                break;

            default:
                console.warn(`AdminActions: Unknown action "${action}"`);
        }
    },

    /**
     * Initialize event delegation
     */
    init() {
        document.addEventListener('click', (e) => this.handleClick(e));

        // Also handle modal close on overlay click and Escape key
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-overlay')) {
                if (typeof AdminAnimations !== 'undefined') {
                    AdminAnimations.closeModalAnimated(e.target);
                } else {
                    e.target.classList.add('hidden');
                }
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                document.querySelectorAll('.modal-overlay:not(.hidden)').forEach(modal => {
                    if (typeof AdminAnimations !== 'undefined') {
                        AdminAnimations.closeModalAnimated(modal);
                    } else {
                        modal.classList.add('hidden');
                    }
                });
            }
        });
    }
};

/**
 * Animation helpers for admin UI
 */
const AdminAnimations = {
    /**
     * Apply staggered entrance delays to a set of elements.
     * @param {string} selector - CSS selector for elements to stagger
     * @param {number} baseDelay - Base delay in ms (default 40)
     */
    staggerEntrance(selector, baseDelay = 40) {
        const items = document.querySelectorAll(selector);
        items.forEach((el, i) => {
            el.style.setProperty('--stagger-delay', `${i * baseDelay}ms`);
        });
    },

    /**
     * Animate a number counting up from 0 to its current text value.
     * @param {HTMLElement} el - Element containing a numeric value
     * @param {number} duration - Animation duration in ms (default 600)
     */
    countUp(el, duration = 600) {
        const text = el.textContent.trim();
        const target = parseFloat(text.replace(/,/g, ''));
        if (isNaN(target) || target === 0) return;

        const isInt = Number.isInteger(target);
        const start = performance.now();

        el.textContent = '0';

        function tick(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = target * eased;

            el.textContent = isInt
                ? Math.round(current).toLocaleString()
                : current.toFixed(1);

            if (progress < 1) {
                requestAnimationFrame(tick);
            } else {
                el.textContent = text; // Restore exact original text
                el.classList.add('counted');
            }
        }

        requestAnimationFrame(tick);
    },

    /**
     * Animate all stat counters on the page.
     */
    animateCounters() {
        const selectors = [
            '.stat__number',
            '.stat-card__value',
            '.status-card__value'
        ];
        selectors.forEach(sel => {
            document.querySelectorAll(sel).forEach(el => {
                // Only animate if the content looks numeric
                const text = el.textContent.trim();
                if (/^\d[\d,.]*$/.test(text)) {
                    this.countUp(el);
                }
            });
        });
    },

    /**
     * Enhanced alert with slide-in + auto-dismiss animation.
     * @param {HTMLElement} container - Alert container
     * @param {string} message - Alert message
     * @param {string} type - 'success' or 'error'
     * @param {number} dismissAfter - Auto dismiss ms (default 4000)
     */
    showAlert(container, message, type, dismissAfter = 4000) {
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} show`;
        alert.textContent = message;
        container.innerHTML = '';
        container.appendChild(alert);

        if (dismissAfter > 0) {
            setTimeout(() => {
                alert.classList.add('dismissing');
                alert.addEventListener('animationend', () => alert.remove(), { once: true });
            }, dismissAfter);
        }
    },

    /**
     * Close a modal with scale-out animation.
     * @param {HTMLElement} overlay - Modal overlay element
     */
    closeModalAnimated(overlay) {
        overlay.classList.add('closing');
        let done = false;
        const finish = () => {
            if (done) return;
            done = true;
            overlay.classList.remove('closing');
            overlay.classList.add('hidden');
            overlay.classList.remove('visible');
        };
        overlay.addEventListener('animationend', finish, { once: true });
        // Fallback: if no close animation fires (missing keyframe, hidden child,
        // reduced-motion), still hide the overlay so it never traps the UI.
        setTimeout(finish, 300);
    },

    /**
     * Initialize all page animations.
     */
    init() {
        // Stagger cards and grid items
        this.staggerEntrance('.admin-card');
        this.staggerEntrance('.stat-card');
        this.staggerEntrance('.job-card');
        this.staggerEntrance('.status-card');

        // Animate stat counters
        this.animateCounters();
    }
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        AdminUtils.init();
        AdminActions.init();
        ThemeToggle.injectButton();
        AdminAnimations.init();
    });
} else {
    AdminUtils.init();
    AdminActions.init();
    ThemeToggle.injectButton();
    AdminAnimations.init();
}
