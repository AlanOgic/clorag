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

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => AdminUtils.init());
} else {
    AdminUtils.init();
}
