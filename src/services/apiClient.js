/**
 * Low-level API fetch utility.
 * All service modules use this for HTTP calls.
 */
import { Platform } from 'react-native';
import Constants from 'expo-constants';
import { getToken } from '../storage/tokenStorage';

/**
 * Get the API base URL based on platform
 * - Web: uses localhost or configured URL
 * - Mobile: requires actual IP address (localhost won't work)
 */
function getApiBaseUrl() {
  const configuredUrl = (process.env.EXPO_PUBLIC_API_BASE_URL || '').trim().replace(/\/$/, '');
  
  // If a URL is configured and it's not localhost, use it
  if (configuredUrl && !configuredUrl.includes('localhost') && !configuredUrl.includes('127.0.0.1')) {
    return configuredUrl;
  }
  
  // For web, localhost works fine
  if (Platform.OS === 'web') {
    return configuredUrl || 'http://localhost:3000';
  }
  
  // For mobile, try to get the dev server host IP from Expo
  const debuggerHost = Constants.expoConfig?.hostUri || Constants.manifest?.debuggerHost;
  if (debuggerHost) {
    // Extract IP from debuggerHost (format: "192.168.x.x:19000")
    const hostIp = debuggerHost.split(':')[0];
    if (hostIp && hostIp !== 'localhost' && hostIp !== '127.0.0.1') {
      return `http://${hostIp}:3000`;
    }
  }
  
  // Fallback - localhost (will fail on physical device but works on emulator with port forwarding)
  return 'http://localhost:3000';
}

const API_BASE_URL = getApiBaseUrl();

// Log the resolved URL for debugging
console.log(`[apiClient] Using API URL: ${API_BASE_URL} (Platform: ${Platform.OS})`);

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
