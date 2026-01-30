/**
 * ML Service - connects to the Python Flask ML backend
 * Provides Talisay fruit analysis with real machine learning predictions
 */

import { Platform } from 'react-native';
import Constants from 'expo-constants';
import * as ImageManipulator from 'expo-image-manipulator';

// Performance settings for image optimization
const IMAGE_CONFIG = {
  // Target dimensions for ML analysis (smaller = faster transfer)
  maxWidth: 1024,
  maxHeight: 1024,
  // JPEG quality (0.6-0.8 is good balance of quality vs size)
  quality: 0.7,
  // Request timeout in milliseconds (30 seconds)
  timeout: 30000,
};

/**
 * Get the ML API URL based on platform
 * - Web: uses localhost or configured URL
 * - Mobile: requires actual IP address (localhost won't work)
 */
function getMLApiUrl() {
  const configuredUrl = (process.env.EXPO_PUBLIC_ML_API_URL || '').trim().replace(/\/$/, '');
  
  // If a URL is configured and it's not localhost, use it
  if (configuredUrl && !configuredUrl.includes('localhost') && !configuredUrl.includes('127.0.0.1')) {
    return configuredUrl;
  }
  
  // For web, localhost works fine
  if (Platform.OS === 'web') {
    return configuredUrl || 'http://localhost:5001';
  }
  
  // For mobile, try to get the dev server host IP from Expo
  const debuggerHost = Constants.expoConfig?.hostUri || Constants.manifest?.debuggerHost;
  if (debuggerHost) {
    // Extract IP from debuggerHost (format: "192.168.x.x:19000")
    const hostIp = debuggerHost.split(':')[0];
    if (hostIp && hostIp !== 'localhost' && hostIp !== '127.0.0.1') {
      return `http://${hostIp}:5001`;
    }
  }
  
  // Fallback - localhost (will fail on physical device but works on emulator with port forwarding)
  return 'http://localhost:5001';
}

// ML Backend URL - dynamically resolved based on platform
const ML_API_URL = getMLApiUrl();

// Log the resolved URL for debugging
console.log(`[mlService] Using ML API URL: ${ML_API_URL} (Platform: ${Platform.OS})`);

/**
 * Check if ML backend is configured and reachable
 */
export async function isMLBackendAvailable() {
  try {
    const response = await fetch(`${ML_API_URL}/`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    const data = await response.json();
    return data.status === 'healthy';
  } catch (error) {
    console.warn('[mlService] ML backend not available:', error.message);
    return false;
  }
}

/**
 * Get ML backend system info
 */
export async function getMLSystemInfo() {
  try {
    const response = await fetch(`${ML_API_URL}/api/info`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    return await response.json();
  } catch (error) {
    console.warn('[mlService] Failed to get system info:', error.message);
    return null;
  }
}

/**
 * Analyze a Talisay fruit image using the ML backend
 * 
 * @param {string} imageUri - Local image URI from camera/gallery
 * @param {object} options - Optional parameters
 * @param {object} options.dimensions - Known dimensions { length_cm, width_cm, kernel_mass_g }
 * @param {function} options.onProgress - Progress callback (stage, message)
 * @returns {Promise<object>} Analysis result from ML backend
 */
export async function analyzeImage(imageUri, options = {}) {
  const { onProgress } = options;
  const startTime = Date.now();
  
  try {
    // Step 1: Optimize image for faster transfer
    onProgress?.('optimizing', 'Optimizing image...');
    console.log('[mlService] Starting image optimization...');
    
    let optimizedUri = imageUri;
    let originalSize = 0;
    let optimizedSize = 0;
    
    // Only optimize on native platforms (web images are usually already small)
    if (Platform.OS !== 'web') {
      try {
        // Resize and compress image
        const manipResult = await ImageManipulator.manipulateAsync(
          imageUri,
          [{ resize: { width: IMAGE_CONFIG.maxWidth, height: IMAGE_CONFIG.maxHeight } }],
          { 
            compress: IMAGE_CONFIG.quality, 
            format: ImageManipulator.SaveFormat.JPEG,
          }
        );
        optimizedUri = manipResult.uri;
        console.log(`[mlService] Image optimized: ${IMAGE_CONFIG.maxWidth}x${IMAGE_CONFIG.maxHeight}, quality: ${IMAGE_CONFIG.quality}`);
      } catch (manipError) {
        console.warn('[mlService] Image optimization failed, using original:', manipError.message);
        // Continue with original image if optimization fails
      }
    }
    
    // Step 2: Convert to base64
    onProgress?.('encoding', 'Preparing image...');
    console.log('[mlService] Converting to base64...');
    
    let base64Image;
    
    try {
      const response = await fetch(optimizedUri);
      const blob = await response.blob();
      originalSize = blob.size;
      base64Image = await blobToBase64(blob);
      optimizedSize = base64Image.length;
      console.log(`[mlService] Image size: ${(originalSize / 1024).toFixed(1)}KB, base64: ${(optimizedSize / 1024).toFixed(1)}KB`);
    } catch (fetchError) {
      console.warn('[mlService] Fetch failed, trying alternative method:', fetchError.message);
      // Fallback: if the URI is already a data URL, extract base64
      if (imageUri.startsWith('data:')) {
        base64Image = imageUri.split(',')[1];
      } else {
        throw new Error('Failed to read image: ' + fetchError.message);
      }
    }

    // Step 3: Send to ML backend with timeout
    onProgress?.('analyzing', 'Analyzing fruit...');
    console.log('[mlService] Sending to ML backend...');

    // Prepare request body
    const requestBody = {
      image: base64Image,
    };

    if (options.dimensions) {
      requestBody.dimensions = options.dimensions;
    }

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), IMAGE_CONFIG.timeout);

    try {
      // Call ML backend with timeout
      const response = await fetch(`${ML_API_URL}/api/predict/image`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const data = await response.json();
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`[mlService] Analysis complete in ${elapsed}s`);

      if (!response.ok) {
        throw new Error(data.error || `ML API error: ${response.status}`);
      }

      if (!data.success) {
        throw new Error(data.error || 'Analysis failed');
      }

      return {
        success: true,
        ...transformMLResult(data.result),
        raw: data.result, // Include raw result for debugging
        timing: {
          totalSeconds: parseFloat(elapsed),
          imageSizeKB: (originalSize / 1024).toFixed(1),
        },
      };
    } catch (fetchError) {
      clearTimeout(timeoutId);
      if (fetchError.name === 'AbortError') {
        throw new Error('Analysis timed out. Please try again or check your connection.');
      }
      throw fetchError;
    }

  } catch (error) {
    console.error('[mlService.analyzeImage]', error);
    return {
      success: false,
      error: error.message,
      fallbackAvailable: true,
    };
  }
}

/**
 * Get prediction from manual measurements (no image required)
 * 
 * @param {object} params - Measurement parameters
 * @param {string} params.color - Fruit color (green, yellow, brown)
 * @param {number} params.lengthCm - Fruit length in cm
 * @param {number} params.widthCm - Fruit width in cm
 * @param {number} params.kernelMassG - Kernel mass in grams (optional)
 * @returns {Promise<object>} Prediction result
 */
export async function predictFromMeasurements({ color, lengthCm, widthCm, kernelMassG }) {
  try {
    const response = await fetch(`${ML_API_URL}/api/predict/measurements`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({
        color: color.toLowerCase(),
        length_cm: lengthCm,
        width_cm: widthCm,
        kernel_mass_g: kernelMassG,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || `ML API error: ${response.status}`);
    }

    return {
      success: true,
      ...transformMLResult(data.result),
      raw: data.result,
    };

  } catch (error) {
    console.error('[mlService.predictFromMeasurements]', error);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Transform ML backend result to app-friendly format
 */
function transformMLResult(result) {
  // Map ML result to app format
  const category = (result.color || 'brown').toUpperCase();
  
  return {
    // Fruit validation
    isTalisay: result.is_talisay !== false,
    fruitValidation: result.fruit_validation || null,
    userMessage: result.user_message || null,
    
    // Color classification
    category,
    color: result.color,
    maturityStage: result.maturity_stage,
    colorConfidence: result.color_confidence,
    colorMethod: result.color_method_used,
    
    // Spot detection
    hasSpots: result.has_spots || false,
    spotCoverage: result.spot_coverage_percent || 0,
    
    // Dimensions
    dimensions: result.dimensions || {},
    dimensionsSource: result.dimensions_source,
    dimensionsConfidence: result.dimensions_confidence,
    referenceDetected: result.reference_detected || false,
    measurementMode: result.measurement_mode,
    coinInfo: result.coin_info || { detected: false },
    measurementTip: result.measurement_tip,
    
    // Oil yield prediction
    oilYieldPercent: result.oil_yield_percent,
    yieldCategory: result.yield_category,
    oilConfidence: result.oil_confidence,
    interpretation: result.interpretation,
    
    // Overall
    overallConfidence: result.overall_confidence,
    analysisComplete: result.analysis_complete,
    
    // Segmentation info
    segmentation: result.segmentation || null,
  };
}

/**
 * Convert blob to base64 (for web platform)
 */
function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64 = reader.result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Get photo guide for best results
 */
export function getPhotoGuide() {
  return {
    title: 'Photo Guide for Best Results',
    coinPlacement: {
      position: 'LEFT side',
      coinType: '₱5 Silver Coin (NEW)',
      diameter: '25mm',
    },
    fruitPlacement: {
      position: 'RIGHT side',
    },
    tips: [
      'Place the ₱5 coin on the LEFT side of the image',
      'Place the Talisay fruit on the RIGHT side',
      'Keep both at the same vertical height',
      'Use a plain background (white, black, or neutral)',
      'Ensure good lighting (avoid harsh shadows)',
      'Take photo from directly above (top-down view)',
      'Fill 60-80% of frame with coin and fruit',
    ],
    withoutCoin: 'System can still estimate dimensions, but results will be less precise.',
  };
}

export default {
  isMLBackendAvailable,
  getMLSystemInfo,
  analyzeImage,
  predictFromMeasurements,
  getPhotoGuide,
};
