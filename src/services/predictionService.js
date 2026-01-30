/**
 * Prediction service - create and list predictions.
 */
import { API_ENDPOINTS } from '../constants';
import { apiFetch, isApiConfigured } from './apiClient';

/**
 * Save a prediction to the backend.
 * @param {object} prediction - { category, confidence, ratio, inputs, imageUri }
 * @returns {Promise<{ id: string } | null>} Created prediction ID or null if API not configured
 */
export async function createPrediction(prediction) {
  if (!isApiConfigured()) {
    return null;
  }

  try {
    const res = await apiFetch(API_ENDPOINTS.PREDICTIONS, {
      method: 'POST',
      body: prediction,
    });
    return { id: res.id };
  } catch (e) {
    console.warn('[predictionService.createPrediction]', e?.message);
    return null;
  }
}

/**
 * List current user's predictions from the backend.
 * @returns {Promise<object[]>} Array of predictions
 */
export async function listPredictions() {
  const res = await apiFetch(API_ENDPOINTS.PREDICTIONS);
  return res.items || [];
}
