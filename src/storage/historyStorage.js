/**
 * AsyncStorage wrapper for local prediction history.
 */
import AsyncStorage from '@react-native-async-storage/async-storage';
import { STORAGE_KEYS, HISTORY_LIMIT } from '../constants';

export async function loadHistory() {
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEYS.HISTORY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    console.warn('[historyStorage.loadHistory]', e);
    return [];
  }
}

export async function saveHistory(items) {
  try {
    await AsyncStorage.setItem(STORAGE_KEYS.HISTORY, JSON.stringify(items));
  } catch (e) {
    console.warn('[historyStorage.saveHistory]', e);
  }
}

export async function addHistoryItem(item) {
  const existing = await loadHistory();
  const next = [item, ...existing].slice(0, HISTORY_LIMIT);
  await saveHistory(next);
  return next;
}

export async function clearHistory() {
  try {
    await AsyncStorage.removeItem(STORAGE_KEYS.HISTORY);
  } catch (e) {
    console.warn('[historyStorage.clearHistory]', e);
  }
}
