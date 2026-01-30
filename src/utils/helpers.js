/**
 * General helper utilities.
 */

/**
 * Parse a numeric input, returning undefined if invalid.
 * @param {string} text
 * @returns {number | undefined}
 */
export function parseNumber(text) {
  const value = Number(String(text).replace(/[^0-9.\-]/g, ''));
  return Number.isFinite(value) ? value : undefined;
}

/**
 * Get a human-readable label for a fruit category.
 * @param {string} category - 'GREEN' | 'YELLOW' | 'BROWN'
 * @returns {string}
 */
export function getCategoryLabel(category) {
  switch (category) {
    case 'GREEN':
      return 'Green (Unripe/Fresh)';
    case 'YELLOW':
      return 'Yellow (Ripe)';
    case 'BROWN':
    default:
      return 'Brown (Overripe/Dry)';
  }
}

/**
 * Format a ratio as a percentage string.
 * @param {number} ratio
 * @returns {string}
 */
export function formatRatioPercent(ratio) {
  return `${Math.round(ratio * 100)}%`;
}
