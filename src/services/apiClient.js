/**
 * Low-level API fetch utility.
 * All service modules use this for HTTP calls.
 */
import { getToken } from '../storage/tokenStorage';

const API_BASE_URL = (process.env.EXPO_PUBLIC_API_BASE_URL || '').trim().replace(/\/$/, '');

function makeUrl(path) {
  if (!API_BASE_URL) {
    throw new Error('Missing EXPO_PUBLIC_API_BASE_URL');
  }
  return `${API_BASE_URL}${path.startsWith('/') ? path : `/${path}`}`;
}

/**
 * Perform an authenticated (or unauthenticated) API request.
 * Automatically attaches Bearer token if available.
 *
 * @param {string} path - API endpoint path (e.g., '/api/auth/login')
 * @param {object} options - { method, body, token, skipAuth }
 * @returns {Promise<object>} Parsed JSON response
 */
export async function apiFetch(path, { method = 'GET', body, token, skipAuth = false } = {}) {
  const resolvedToken = skipAuth ? null : (token ?? (await getToken()));

  const headers = {
    'Content-Type': 'application/json',
  };

  if (resolvedToken) {
    headers['Authorization'] = `Bearer ${resolvedToken}`;
  }

  const res = await fetch(makeUrl(path), {
    method,
    headers,
    ...(body ? { body: JSON.stringify(body) } : null),
  });

  const text = await res.text();
  let json = null;

  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = null;
  }

  if (!res.ok) {
    const msg = json?.error || `http_${res.status}`;
    const err = new Error(msg);
    err.status = res.status;
    err.payload = json;
    throw err;
  }

  return json;
}

/**
 * Check if the API base URL is configured.
 */
export function isApiConfigured() {
  return Boolean(API_BASE_URL);
}
