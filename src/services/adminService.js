/**
 * Admin service - admin-only endpoints.
 */
import { API_ENDPOINTS } from '../constants';
import { apiFetch } from './apiClient';

/**
 * List all users (admin only).
 * @returns {Promise<object[]>} Array of users
 */
export async function listUsers() {
  const res = await apiFetch(API_ENDPOINTS.ADMIN.USERS);
  return res.users || [];
}

/**
 * List all predictions (admin only).
 * @returns {Promise<object[]>} Array of predictions
 */
export async function listAllPredictions() {
  const res = await apiFetch(API_ENDPOINTS.ADMIN.PREDICTIONS);
  return res.items || [];
}
