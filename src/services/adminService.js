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

/**
 * List all history from all users (admin only).
 * @param {number} limit - Max items to return
 * @returns {Promise<object[]>} Array of history items
 */
export async function listAllHistory(limit = 100) {
  const res = await apiFetch(`${API_ENDPOINTS.ADMIN.HISTORY}?limit=${limit}`);
  return res.items || [];
}

/**
 * Get comprehensive analytics overview (admin only).
 * @returns {Promise<object>} Analytics data
 */
export async function getAnalyticsOverview() {
  const res = await apiFetch(API_ENDPOINTS.ADMIN.ANALYTICS_OVERVIEW);
  return res.analytics || {};
}

/**
 * Get chart-specific data (admin only).
 * @param {string} chartType - Type of chart data to fetch
 * @returns {Promise<object[]>} Chart data
 */
export async function getChartData(chartType) {
  const res = await apiFetch(`${API_ENDPOINTS.ADMIN.ANALYTICS_CHARTS}?chartType=${chartType}`);
  return res.data || [];
}
