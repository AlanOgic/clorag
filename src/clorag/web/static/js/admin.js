/**
 * Admin utilities for CLORAG
 * Provides CSRF protection and common admin functionality
 */

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
     * Close a modal
     */
    closeModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
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
     * Call a global function by name
     */
    callFunction(fnName, args) {
        const fn = window[fnName];
        if (typeof fn === 'function') {
            const parsedArgs = args ? args.split(',').map(a => {
                const trimmed = a.trim();
                // Try to parse as number
                if (/^\d+$/.test(trimmed)) return parseInt(trimmed, 10);
                if (/^\d+\.\d+$/.test(trimmed)) return parseFloat(trimmed);
                // Remove quotes if present
                if (/^['"].*['"]$/.test(trimmed)) return trimmed.slice(1, -1);
                return trimmed;
            }) : [];
            fn(...parsedArgs);
        } else {
            console.warn(`AdminActions: Function "${fnName}" not found`);
        }
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
                this.callFunction(actionEl.dataset.fn, actionEl.dataset.args);
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
                e.target.classList.add('hidden');
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                document.querySelectorAll('.modal-overlay:not(.hidden)').forEach(modal => {
                    modal.classList.add('hidden');
                });
            }
        });
    }
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        AdminUtils.init();
        AdminActions.init();
    });
} else {
    AdminUtils.init();
    AdminActions.init();
}
