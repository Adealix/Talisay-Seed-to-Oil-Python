/**
 * AuthContext - Global authentication state management.
 */
import React, { createContext, useCallback, useEffect, useMemo, useState } from 'react';
import { authService } from '../services';
import { getToken, setToken, clearToken } from '../storage';

export const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setTokenState] = useState(null);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // ---------------------------------------------------------------------------
  // Refresh current user from server
  // ---------------------------------------------------------------------------
  const refreshMe = useCallback(async (maybeToken) => {
    const t = maybeToken ?? token;
    if (!t) {
      setUser(null);
      return;
    }

    try {
      const userData = await authService.getMe(t);
      setUser(userData);
    } catch (e) {
      console.warn('[AuthContext.refreshMe]', e?.message);
      setUser(null);
    }
  }, [token]);

  // ---------------------------------------------------------------------------
  // Restore token on mount
  // ---------------------------------------------------------------------------
  useEffect(() => {
    let mounted = true;

    (async () => {
      try {
        const stored = await getToken();
        if (!mounted) return;

        setTokenState(stored);
        if (stored) {
          await refreshMe(stored);
        }
      } catch (e) {
        console.warn('[AuthContext.restore]', e);
      } finally {
        if (mounted) setLoading(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, [refreshMe]);

  // ---------------------------------------------------------------------------
  // Auth actions
  // ---------------------------------------------------------------------------
  const login = useCallback(async ({ email, password }) => {
    const { token: newToken, user: userData } = await authService.login({ email, password });
    await setToken(newToken);
    setTokenState(newToken);
    setUser(userData);
  }, []);

  const register = useCallback(async ({ email, password, role }) => {
    const { token: newToken, user: userData } = await authService.register({ email, password, role });
    await setToken(newToken);
    setTokenState(newToken);
    setUser(userData);
  }, []);

  const logout = useCallback(async () => {
    await clearToken();
    setTokenState(null);
    setUser(null);
  }, []);

  // ---------------------------------------------------------------------------
  // Context value
  // ---------------------------------------------------------------------------
  const value = useMemo(
    () => ({
      token,
      user,
      loading,
      isAuthenticated: Boolean(user),
      isAdmin: user?.role === 'admin',
      login,
      register,
      logout,
      refreshMe,
    }),
    [token, user, loading, login, register, logout, refreshMe]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
