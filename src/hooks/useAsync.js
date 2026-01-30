/**
 * useAsync hook - manage async operation state.
 */
import { useCallback, useState } from 'react';

/**
 * @returns {{ loading, error, run, reset }}
 */
export function useAsync() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const run = useCallback(async (asyncFn) => {
    setError(null);
    setLoading(true);
    try {
      const result = await asyncFn();
      return result;
    } catch (e) {
      setError(e?.message || 'An error occurred');
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setError(null);
    setLoading(false);
  }, []);

  return { loading, error, run, reset };
}
