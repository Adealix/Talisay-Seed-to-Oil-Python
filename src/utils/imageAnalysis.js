/**
 * Image analysis utilities - color detection and ratio prediction.
 */
import * as ImageManipulator from 'expo-image-manipulator';
import { toByteArray } from 'base64-js';
import UPNG from 'upng-js';
import { FRUIT_CATEGORIES } from '../constants';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function rgbToHsv(r, g, b) {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;

  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;

  let h = 0;
  if (delta !== 0) {
    if (max === rn) h = ((gn - bn) / delta) % 6;
    else if (max === gn) h = (bn - rn) / delta + 2;
    else h = (rn - gn) / delta + 4;
    h *= 60;
    if (h < 0) h += 360;
  }

  const s = max === 0 ? 0 : delta / max;
  const v = max;

  return { h, s, v };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Predict seed-to-oil ratio based on category and optional morphology inputs.
 * This is a heuristic prototype, NOT a real ML model.
 *
 * @param {{ category: string, lengthMm?: number, widthMm?: number, weightG?: number }} params
 * @returns {number} Estimated ratio (0.08 - 0.45)
 */
export function predictRatio({ category, lengthMm, widthMm, weightG }) {
  const baseRatios = {
    [FRUIT_CATEGORIES.GREEN]: 0.35,
    [FRUIT_CATEGORIES.YELLOW]: 0.25,
    [FRUIT_CATEGORIES.BROWN]: 0.15,
  };

  const base = baseRatios[category] ?? 0.15;

  const length = Number.isFinite(lengthMm) ? lengthMm : 0;
  const width = Number.isFinite(widthMm) ? widthMm : 0;
  const weight = Number.isFinite(weightG) ? weightG : 0;

  // Heuristic: larger/heavier fruits slightly increase predicted ratio.
  const sizeTerm = clamp(((length * width) / 2500) * 0.03, 0, 0.03);
  const weightTerm = clamp((weight - 20) / 300, -0.03, 0.04);

  return clamp(base + sizeTerm + weightTerm, 0.08, 0.45);
}

/**
 * Analyze a fruit image to detect its color category.
 * Uses average color → HSV → category heuristic.
 *
 * @param {string} imageUri - Local image URI
 * @returns {Promise<{ category: string, confidence: number, averageRgb: { r, g, b } }>}
 */
export async function analyzeFruitImage(imageUri) {
  // Resize to tiny image for fast PNG decode.
  const manipulated = await ImageManipulator.manipulateAsync(
    imageUri,
    [{ resize: { width: 64 } }],
    { compress: 1, format: ImageManipulator.SaveFormat.PNG, base64: true }
  );

  if (!manipulated.base64) {
    throw new Error('Image processing failed: missing base64.');
  }

  const bytes = toByteArray(manipulated.base64);
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

  const img = UPNG.decode(buffer);
  const rgbaFrames = UPNG.toRGBA8(img);

  if (!rgbaFrames || rgbaFrames.length === 0) {
    throw new Error('PNG decode failed.');
  }

  const rgba = new Uint8Array(rgbaFrames[0]);
  const pixelCount = rgba.length / 4;

  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  let counted = 0;

  // Sample every Nth pixel to reduce cost.
  const step = Math.max(1, Math.floor(pixelCount / 2048));

  for (let i = 0; i < pixelCount; i += step) {
    const idx = i * 4;
    const a = rgba[idx + 3] / 255;
    if (a < 0.3) continue;

    rSum += rgba[idx];
    gSum += rgba[idx + 1];
    bSum += rgba[idx + 2];
    counted += 1;
  }

  if (counted === 0) {
    throw new Error('No visible pixels found.');
  }

  const rAvg = rSum / counted;
  const gAvg = gSum / counted;
  const bAvg = bSum / counted;

  const { h, s, v } = rgbToHsv(rAvg, gAvg, bAvg);

  // Classify based on hue.
  let category = FRUIT_CATEGORIES.BROWN;

  if (s > 0.18 && v > 0.25) {
    if (h >= 70 && h <= 170) {
      category = FRUIT_CATEGORIES.GREEN;
    } else if (h >= 35 && h < 70) {
      category = FRUIT_CATEGORIES.YELLOW;
    } else if (h >= 15 && h < 35) {
      category = FRUIT_CATEGORIES.BROWN;
    }
  }

  // Confidence heuristic based on saturation and value.
  const confidence = clamp(s * 0.8 + v * 0.2, 0, 1);

  return {
    category,
    confidence,
    averageRgb: {
      r: Math.round(rAvg),
      g: Math.round(gAvg),
      b: Math.round(bAvg),
    },
  };
}
