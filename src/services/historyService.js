/**
 * History service - MongoDB-backed history operations.
 */
import { API_ENDPOINTS } from '../constants';
import { apiFetch, isApiConfigured } from './apiClient';

/**
 * Save a history item to the backend.
 * @param {object} historyItem - History data to save
 * @returns {Promise<{ id: string } | null>} Created history ID or null if failed
 */
export async function saveHistoryItem(historyItem) {
  if (!isApiConfigured()) {
    return null;
  }

  try {
    const res = await apiFetch(API_ENDPOINTS.HISTORY.BASE, {
      method: 'POST',
      body: historyItem,
    });
    return { id: res.id };
  } catch (e) {
    console.warn('[historyService.saveHistoryItem]', e?.message);
    return null;
  }
}

/**
 * List current user's history from the backend.
 * @param {object} options - { limit, skip }
 * @returns {Promise<{ items: object[], total: number, hasMore: boolean }>}
 */
export async function listHistory({ limit = 50, skip = 0 } = {}) {
  try {
    const res = await apiFetch(`${API_ENDPOINTS.HISTORY.BASE}?limit=${limit}&skip=${skip}`);
    return {
      items: res.items || [],
      total: res.total || 0,
      hasMore: res.hasMore || false,
    };
  } catch (e) {
    console.warn('[historyService.listHistory]', e?.message);
    return { items: [], total: 0, hasMore: false };
  }
}

/**
 * Get a single history item by ID.
 * @param {string} id - History item ID
 * @returns {Promise<object | null>}
 */
export async function getHistoryItem(id) {
  try {
    const res = await apiFetch(API_ENDPOINTS.HISTORY.ITEM(id));
    return res.item || null;
  } catch (e) {
    console.warn('[historyService.getHistoryItem]', e?.message);
    return null;
  }
}

/**
 * Delete a history item by ID.
 * @param {string} id - History item ID
 * @returns {Promise<boolean>}
 */
export async function deleteHistoryItem(id) {
  try {
    await apiFetch(API_ENDPOINTS.HISTORY.ITEM(id), { method: 'DELETE' });
    return true;
  } catch (e) {
    console.warn('[historyService.deleteHistoryItem]', e?.message);
    return false;
  }
}

/**
 * Clear all history for the current user.
 * @returns {Promise<{ deletedCount: number }>}
 */
export async function clearAllHistory() {
  try {
    const res = await apiFetch(API_ENDPOINTS.HISTORY.BASE, { method: 'DELETE' });
    return { deletedCount: res.deletedCount || 0 };
  } catch (e) {
    console.warn('[historyService.clearAllHistory]', e?.message);
    return { deletedCount: 0 };
  }
}
