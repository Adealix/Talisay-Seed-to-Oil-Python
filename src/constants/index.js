/**
 * App-wide constants
 */

export const STORAGE_KEYS = {
  AUTH_TOKEN: 'talisay_auth_token_v1',
  HISTORY: 'talisay_ml_history_v1',
};

export const API_ENDPOINTS = {
  AUTH: {
    LOGIN: '/api/auth/login',
    REGISTER: '/api/auth/register',
    ME: '/api/auth/me',
  },
  PREDICTIONS: '/api/predictions',
  HISTORY: {
    BASE: '/api/history',
    ITEM: (id) => `/api/history/${id}`,
  },
  ADMIN: {
    USERS: '/api/admin/users',
    PREDICTIONS: '/api/admin/predictions',
    HISTORY: '/api/admin/history',
    ANALYTICS_OVERVIEW: '/api/admin/analytics/overview',
    ANALYTICS_CHARTS: '/api/admin/analytics/charts',
  },
};

export const USER_ROLES = {
  USER: 'user',
  ADMIN: 'admin',
};

export const FRUIT_CATEGORIES = {
  GREEN: 'GREEN',
  YELLOW: 'YELLOW',
  BROWN: 'BROWN',
};

export const HISTORY_LIMIT = 50;
