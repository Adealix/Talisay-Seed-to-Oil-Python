/**
 * useForm hook - simple form state management.
 */
import { useCallback, useState } from 'react';

/**
 * @param {object} initialValues - Initial form field values
 * @returns {{ values, setField, reset, setValues }}
 */
export function useForm(initialValues = {}) {
  const [values, setValues] = useState(initialValues);

  const setField = useCallback((field, value) => {
    setValues((prev) => ({ ...prev, [field]: value }));
  }, []);

  const reset = useCallback(() => {
    setValues(initialValues);
  }, [initialValues]);

  return { values, setField, reset, setValues };
}
