/**
 * AsyncStorage wrapper for auth token persistence.
 */
import AsyncStorage from '@react-native-async-storage/async-storage';
import { STORAGE_KEYS } from '../constants';

export async function getToken() {
  try {
    return await AsyncStorage.getItem(STORAGE_KEYS.AUTH_TOKEN);
  } catch (e) {
    console.warn('[tokenStorage.getToken]', e);
    return null;
  }
}

export async function setToken(token) {
  try {
    if (!token) {
      await AsyncStorage.removeItem(STORAGE_KEYS.AUTH_TOKEN);
    } else {
      await AsyncStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, token);
    }
  } catch (e) {
    console.warn('[tokenStorage.setToken]', e);
  }
}

export async function clearToken() {
  return setToken(null);
}
