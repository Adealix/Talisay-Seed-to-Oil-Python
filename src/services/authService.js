/**
 * Auth service - login, register, get current user.
 */
import { API_ENDPOINTS } from '../constants';
import { apiFetch } from './apiClient';

/**
 * Register a new user account.
 * @param {{ email: string, password: string, role?: 'user' | 'admin' }} data
 * @returns {Promise<{ token: string, user: object }>}
 */
export async function register({ email, password, role = 'user' }) {
  const res = await apiFetch(API_ENDPOINTS.AUTH.REGISTER, {
    method: 'POST',
    body: { email, password, role },
    skipAuth: true,
  });
  return { token: res.token, user: res.user };
}

/**
 * Login with email and password.
 * @param {{ email: string, password: string }} data
 * @returns {Promise<{ token: string, user: object }>}
 */
export async function login({ email, password }) {
  const res = await apiFetch(API_ENDPOINTS.AUTH.LOGIN, {
    method: 'POST',
    body: { email, password },
    skipAuth: true,
  });
  return { token: res.token, user: res.user };
}

/**
 * Get the current authenticated user.
 * @param {string} [token] - Optional token override
 * @returns {Promise<object>} User object
 */
export async function getMe(token) {
  const res = await apiFetch(API_ENDPOINTS.AUTH.ME, { token });
  return res.user;
}
