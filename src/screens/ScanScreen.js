import React, { useState, useEffect } from 'react';
import { Image, Platform, Pressable, StyleSheet, Text, View, ActivityIndicator, Alert, ScrollView, Modal, useWindowDimensions } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import Screen from '../components/Screen';
import { Divider } from '../components/Ui';
import { addHistoryItem } from '../storage';
import { predictionService, mlService, historyService } from '../services';
import { useAuth } from '../hooks';
import { theme } from '../theme/theme';

// Breakpoint for mobile/tablet
const MOBILE_BREAKPOINT = 600;

// Extract filename from URI
function getFilename(uri) {
  if (!uri) return null;
  const parts = uri.split('/');
  let filename = parts[parts.length - 1];
  // Remove query parameters if any
  if (filename.includes('?')) {
    filename = filename.split('?')[0];
  }
  return filename;
}

export default function ScanScreen({ navigation }) {
  const { user } = useAuth();
  const { width: screenWidth } = useWindowDimensions();
  const isMobile = screenWidth < MOBILE_BREAKPOINT;
  
  const [imageUri, setImageUri] = useState(null);
  const [imageName, setImageName] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeProgress, setAnalyzeProgress] = useState(''); // Progress message
  const [mlResult, setMlResult] = useState(null);
  const [error, setError] = useState(null);
  const [mlBackendAvailable, setMlBackendAvailable] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    checkMLBackend();
  }, []);

  async function checkMLBackend() {
    const available = await mlService.isMLBackendAvailable();
    setMlBackendAvailable(available);
  }

  async function pickFromGallery() {
    setError(null);
    setMlResult(null);

    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (!perm.granted) {
      setError('Gallery permission required');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({ 
      mediaTypes: ['images'], 
      quality: 0.7 
    });

    if (result.canceled) return;
    const asset = result.assets?.[0];
    if (!asset?.uri) return;

    // Get original filename from asset or extract from URI
    const originalName = asset.fileName || getFilename(asset.uri);
    await processImage(asset.uri, originalName);
  }

  async function takePhoto() {
    setError(null);
    setMlResult(null);

    // Camera not supported on web
    if (Platform.OS === 'web') {
      setError('Camera not supported on web. Use Gallery instead.');
      return;
    }

    const perm = await ImagePicker.requestCameraPermissionsAsync();
    
    if (!perm.granted) {
      setError('Camera permission required. Please allow camera access in settings.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({ 
      mediaTypes: ['images'], 
      quality: 0.7,
      allowsEditing: false,
    });

    if (result.canceled) return;
    const asset = result.assets?.[0];
    if (!asset?.uri) return;

    // For camera photos, use timestamp-based name
    const photoName = `photo_${Date.now()}.jpg`;
    await processImage(asset.uri, photoName);
  }

  async function processImage(uri, filename) {
    setImageUri(uri);
    setImageName(filename || 'Unknown');
    setAnalyzing(true);
    setAnalyzeProgress('Starting...');

    try {
      if (!mlBackendAvailable) {
        setError('ML Backend offline. Run: cd ml && python api.py');
        setAnalyzing(false);
        setAnalyzeProgress('');
        return;
      }

      // Progress callback for user feedback
      const onProgress = (stage, message) => {
        setAnalyzeProgress(message);
      };

      const mlAnalysis = await mlService.analyzeImage(uri, { onProgress });
      
      if (mlAnalysis.success) {
        setMlResult(mlAnalysis);
        if (!mlAnalysis.isTalisay) {
          setError('Not a Talisay fruit. Try another image.');
        }
      } else {
        setError(mlAnalysis.error || 'Analysis failed');
      }
    } catch (e) {
      setError('Analysis failed. Check ML backend.');
    } finally {
      setAnalyzing(false);
      setAnalyzeProgress('');
    }
  }

  async function saveResult() {
    if (!mlResult || saving) return;

    // Check if user is logged in
    if (!user) {
      setShowLoginModal(true);
      return;
    }

    setSaving(true);

    try {
      const item = {
        id: String(Date.now()),
        createdAt: new Date().toISOString(),
        imageUri,
        imageName: imageName || 'Unknown',
        category: mlResult.category,
        confidence: mlResult.overallConfidence,
        ratio: mlResult.oilYieldPercent / 100,
        mlData: {
          oilYieldPercent: mlResult.oilYieldPercent,
          yieldCategory: mlResult.yieldCategory,
          maturityStage: mlResult.maturityStage,
          dimensions: mlResult.dimensions,
          referenceDetected: mlResult.referenceDetected,
          coinInfo: mlResult.coinInfo,
          interpretation: mlResult.interpretation,
        },
        inputs: {
          lengthMm: mlResult.dimensions?.length_cm ? mlResult.dimensions.length_cm * 10 : null,
          widthMm: mlResult.dimensions?.width_cm ? mlResult.dimensions.width_cm * 10 : null,
          weightG: mlResult.dimensions?.whole_fruit_weight_g ?? null,
        },
      };

      // Save to MongoDB (primary storage)
      const mongoResult = await historyService.saveHistoryItem({
        imageName: item.imageName,
        imageUri: item.imageUri,
        category: item.category,
        maturityStage: mlResult.maturityStage,
        confidence: item.confidence,
        colorConfidence: mlResult.colorConfidence,
        fruitConfidence: mlResult.fruitConfidence,
        oilConfidence: mlResult.oilConfidence,
        // Color probabilities
        colorProbabilities: mlResult.raw?.color_probabilities ? {
          green: mlResult.raw.color_probabilities.green,
          yellow: mlResult.raw.color_probabilities.yellow,
          brown: mlResult.raw.color_probabilities.brown,
        } : null,
        // Spot detection
        hasSpots: mlResult.hasSpots || false,
        spotCoverage: mlResult.spotCoverage || null,
        // Oil yield
        oilYieldPercent: mlResult.oilYieldPercent,
        yieldCategory: mlResult.yieldCategory,
        // Dimensions
        dimensions: {
          lengthCm: mlResult.dimensions?.length_cm ?? null,
          widthCm: mlResult.dimensions?.width_cm ?? null,
          wholeFruitWeightG: mlResult.dimensions?.whole_fruit_weight_g ?? null,
          kernelWeightG: mlResult.dimensions?.kernel_mass_g ?? mlResult.dimensions?.kernel_weight_g ?? null,
        },
        dimensionsSource: mlResult.dimensionsSource || null,
        // Reference detection
        referenceDetected: mlResult.referenceDetected,
        coinInfo: mlResult.coinInfo,
        // Interpretation
        interpretation: mlResult.interpretation,
        // Full analysis data for future reference
        fullAnalysis: {
          colorMethod: mlResult.colorMethod,
          measurementMode: mlResult.measurementMode,
          measurementTip: mlResult.measurementTip,
          dimensionsConfidence: mlResult.dimensionsConfidence,
          segmentation: mlResult.segmentation,
        },
      });

      if (mongoResult?.id) {
        item.mongoId = mongoResult.id;
        console.log('[ScanScreen] Saved to MongoDB:', mongoResult.id);
      } else {
        console.warn('[ScanScreen] MongoDB save returned null - saving to local only');
      }

      // Also save to local storage as backup
      await addHistoryItem(item);
      console.log('[ScanScreen] Saved to local storage');

      // Show success feedback
      if (Platform.OS === 'web') {
        // Navigate immediately on web
        navigation.navigate('History');
      } else {
        Alert.alert('Saved!', 'Prediction saved to history.', [
          { text: 'OK', onPress: () => navigation.navigate('History') }
        ]);
      }
    } catch (e) {
      console.error('[ScanScreen] Save error:', e);
      if (Platform.OS === 'web') {
        alert('Failed to save: ' + (e?.message || 'Unknown error'));
      } else {
        Alert.alert('Save Failed', e?.message || 'Could not save prediction. Please try again.');
      }
    } finally {
      setSaving(false);
    }
  }

  function resetScan() {
    setImageUri(null);
    setImageName(null);
    setMlResult(null);
    setError(null);
    setShowDetails(false);
  }

  const getCategoryColor = (cat) => {
    switch (cat?.toUpperCase()) {
      case 'GREEN': return '#4caf50';
      case 'YELLOW': return '#ffc107';
      case 'BROWN': return '#8b4513';
      default: return '#999';
    }
  };

  // Get color probability bar width
  const getColorBarWidth = (category, current) => {
    if (category?.toUpperCase() === current) {
      return Math.round((mlResult?.colorConfidence || 0) * 100);
    }
    return 0;
  };

  return (
    <Screen scroll={false}>
      {/* Login Required Modal */}
      <Modal
        visible={showLoginModal}
        transparent
        animationType="fade"
        onRequestClose={() => setShowLoginModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <View style={styles.modalIconWrap}>
                <Ionicons name="lock-closed" size={32} color={theme.colors.green} />
              </View>
              <Text style={styles.modalTitle}>Login Required</Text>
              <Text style={styles.modalSubtitle}>You need to login to save prediction history to the cloud.</Text>
            </View>
            <View style={styles.modalBtnRow}>
              <Pressable style={styles.modalBtnCancel} onPress={() => setShowLoginModal(false)}>
                <Text style={styles.modalBtnCancelText}>Cancel</Text>
              </Pressable>
              <Pressable 
                style={styles.modalBtnLogin} 
                onPress={() => {
                  setShowLoginModal(false);
                  navigation.navigate('Login');
                }}
              >
                <Ionicons name="log-in" size={16} color="#fff" />
                <Text style={styles.modalBtnLoginText}>Login</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>

      {/* Compact Header */}
      <View style={[styles.header, isMobile && styles.headerMobile]}>
        <View style={styles.headerLeft}>
          <Ionicons name="scan" size={isMobile ? 18 : 20} color="#fff" />
          <Text style={[styles.headerTitle, isMobile && styles.headerTitleMobile]}>Scan Fruit</Text>
        </View>
        <Pressable onPress={checkMLBackend} style={[styles.statusBadge, { backgroundColor: mlBackendAvailable ? '#4caf50' : '#f44336' }]}>
          <Ionicons name={mlBackendAvailable ? "cloud-done" : "cloud-offline"} size={12} color="#fff" />
          <Text style={styles.statusText}>{mlBackendAvailable ? 'ML Online' : 'Offline'}</Text>
        </Pressable>
      </View>

      {/* Main Content - Responsive Layout */}
      <View style={[styles.mainContent, isMobile && styles.mainContentMobile]}>
        {/* Left Column - Image (on mobile: top section) */}
        <View style={[styles.leftColumn, isMobile && styles.leftColumnMobile]}>
          {/* Button Row */}
          <View style={styles.btnRow}>
            <Pressable style={[styles.pickBtn, isMobile && styles.pickBtnMobile]} onPress={pickFromGallery} disabled={!mlBackendAvailable || analyzing}>
              <Ionicons name="images" size={isMobile ? 20 : 16} color="#fff" />
              <Text style={[styles.pickBtnText, isMobile && styles.pickBtnTextMobile]}>Gallery</Text>
            </Pressable>
            <Pressable 
              style={[styles.pickBtn, styles.pickBtnAlt, isMobile && styles.pickBtnMobile, Platform.OS === 'web' && styles.btnDisabled]} 
              onPress={takePhoto} 
              disabled={!mlBackendAvailable || analyzing || Platform.OS === 'web'}
            >
              <Ionicons name="camera" size={isMobile ? 20 : 16} color={Platform.OS === 'web' ? '#999' : theme.colors.green} />
              <Text style={[styles.pickBtnText, isMobile && styles.pickBtnTextMobile, { color: Platform.OS === 'web' ? '#999' : theme.colors.green }]}>
                {Platform.OS === 'web' ? 'N/A' : 'Camera'}
              </Text>
            </Pressable>
          </View>

          {/* Image Preview */}
          <View style={[styles.imageBox, isMobile && styles.imageBoxMobile]}>
            {imageUri ? (
              <>
                <Image source={{ uri: imageUri }} style={styles.image} />
                {analyzing && (
                  <View style={styles.imageOverlay}>
                    <ActivityIndicator size={isMobile ? "large" : "small"} color="#fff" />
                    <Text style={[styles.overlayText, isMobile && styles.overlayTextMobile]}>
                      {analyzeProgress || 'Analyzing...'}
                    </Text>
                  </View>
                )}
              </>
            ) : (
              <View style={styles.placeholder}>
                <Ionicons name="leaf" size={isMobile ? 48 : 32} color="#ccc" />
                <Text style={[styles.placeholderText, isMobile && styles.placeholderTextMobile]}>Select image</Text>
              </View>
            )}
          </View>

          {/* Filename Display */}
          {imageName && (
            <View style={[styles.filenameBox, isMobile && styles.filenameBoxMobile]}>
              <Ionicons name="document" size={isMobile ? 16 : 12} color={theme.colors.green} />
              <Text style={[styles.filenameText, isMobile && styles.filenameTextMobile]} numberOfLines={1}>{imageName}</Text>
            </View>
          )}

          {/* Photo Tip - Compact */}
          <View style={[styles.tipBox, isMobile && styles.tipBoxMobile]}>
            <Ionicons name="bulb" size={isMobile ? 16 : 12} color="#FFA000" />
            <Text style={[styles.tipText, isMobile && styles.tipTextMobile]}>₱5 coin LEFT • fruit RIGHT</Text>
          </View>

          {error && (
            <View style={[styles.errorBox, isMobile && styles.errorBoxMobile]}>
              <Ionicons name="alert-circle" size={isMobile ? 18 : 14} color="#f44336" />
              <Text style={[styles.errorText, isMobile && styles.errorTextMobile]}>{error}</Text>
            </View>
          )}
        </View>

        {/* Right Column - Results (on mobile: bottom section) */}
        <View style={[styles.rightColumn, isMobile && styles.rightColumnMobile]}>
          {mlResult && mlResult.isTalisay ? (
            <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: isMobile ? 16 : 8 }}>
              {/* Oil Yield */}
              <View style={styles.resultMain}>
                <Text style={[styles.bigPercent, isMobile && styles.bigPercentMobile]}>{Math.round(mlResult.oilYieldPercent || 0)}%</Text>
                <Text style={[styles.resultLabel, isMobile && styles.resultLabelMobile]}>Oil Yield</Text>
              </View>

              {/* Category Badge */}
              <View style={[styles.catBadge, isMobile && styles.catBadgeMobile, { backgroundColor: getCategoryColor(mlResult.category) }]}>
                <Text style={[styles.catText, isMobile && styles.catTextMobile]}>{mlResult.category || 'Unknown'}</Text>
                <Text style={styles.catConf}>{Math.round((mlResult.colorConfidence || 0) * 100)}%</Text>
              </View>

              {/* Expand/Collapse Details Button */}
              <Pressable 
                style={styles.expandBtn} 
                onPress={() => setShowDetails(!showDetails)}
              >
                <Ionicons name={showDetails ? "chevron-up" : "chevron-down"} size={14} color={theme.colors.green} />
                <Text style={styles.expandBtnText}>{showDetails ? 'Hide Details' : 'Show Details'}</Text>
              </Pressable>

              {/* Expanded Details Section */}
              {showDetails && (
                <View style={styles.detailsSection}>
                  {/* Color Classification */}
                  <View style={styles.detailCard}>
                    <View style={styles.detailCardHeader}>
                      <Ionicons name="color-palette" size={14} color={theme.colors.green} />
                      <Text style={styles.detailCardTitle}>Color Classification</Text>
                    </View>
                    <Text style={styles.detailText}>
                      <Text style={styles.detailLabel}>Detected: </Text>
                      <Text style={[styles.detailValue, { color: getCategoryColor(mlResult.category) }]}>
                        {mlResult.category}
                      </Text>
                    </Text>
                    <Text style={styles.detailText}>
                      <Text style={styles.detailLabel}>Confidence: </Text>
                      <Text style={styles.detailValue}>{Math.round((mlResult.colorConfidence || 0) * 100)}%</Text>
                    </Text>
                    <Text style={styles.detailText}>
                      <Text style={styles.detailLabel}>Maturity: </Text>
                      <Text style={styles.detailValue}>{mlResult.maturityStage || 'Unknown'}</Text>
                    </Text>
                    
                    {/* Spot Detection Info */}
                    {mlResult.hasSpots && (
                      <View style={styles.spotWarning}>
                        <Ionicons name="alert-circle" size={14} color="#ff9800" />
                        <Text style={styles.spotWarningText}>
                          ⚠️ Fruit has visible spots ({mlResult.spotCoverage?.toFixed(1) || 0}% coverage)
                        </Text>
                      </View>
                    )}
                    {mlResult.hasSpots && (
                      <Text style={styles.spotNote}>Spots excluded from color analysis for accuracy</Text>
                    )}
                    
                    {/* Color Probabilities */}
                    <Text style={styles.detailSubheader}>Color Probabilities:</Text>
                    {['GREEN', 'YELLOW', 'BROWN'].map(color => {
                      // Try to get probabilities from raw data if available
                      const rawProbs = mlResult.raw?.color_probabilities;
                      let prob = 0;
                      if (rawProbs) {
                        prob = Math.round((rawProbs[color.toLowerCase()] || 0) * 100);
                      } else {
                        // Fallback: show confidence only for active color
                        const isActive = mlResult.category?.toUpperCase() === color;
                        prob = isActive ? Math.round((mlResult.colorConfidence || 0) * 100) : 0;
                      }
                      return (
                        <View key={color} style={styles.probRow}>
                          <Text style={[styles.probLabel, { color: getCategoryColor(color) }]}>{color}</Text>
                          <View style={styles.probBarWrap}>
                            <View style={[styles.probBar, { width: `${prob}%`, backgroundColor: getCategoryColor(color) }]} />
                          </View>
                          <Text style={styles.probValue}>{prob}%</Text>
                        </View>
                      );
                    })}
                  </View>

                  {/* Reference Object */}
                  <View style={styles.detailCard}>
                    <View style={styles.detailCardHeader}>
                      <Ionicons name="disc" size={14} color="#ff9800" />
                      <Text style={styles.detailCardTitle}>Reference Object</Text>
                    </View>
                    {mlResult.referenceDetected ? (
                      <>
                        <View style={styles.detailSuccess}>
                          <Ionicons name="checkmark-circle" size={14} color="#4caf50" />
                          <Text style={styles.detailSuccessText}>Coin detected</Text>
                        </View>
                        <Text style={styles.detailText}>
                          <Text style={styles.detailLabel}>Type: </Text>
                          <Text style={styles.detailValue}>
                            {typeof mlResult.coinInfo === 'object' 
                              ? (mlResult.coinInfo?.coin_name || '₱5 Silver Coin') 
                              : (mlResult.coinInfo || '₱5 Silver Coin (25mm)')}
                          </Text>
                        </Text>
                        {typeof mlResult.coinInfo === 'object' && mlResult.coinInfo?.coin_diameter_cm && (
                          <Text style={styles.detailText}>
                            <Text style={styles.detailLabel}>Diameter: </Text>
                            <Text style={styles.detailValue}>{mlResult.coinInfo.coin_diameter_cm} cm</Text>
                          </Text>
                        )}
                      </>
                    ) : (
                      <>
                        <View style={styles.detailWarning}>
                          <Ionicons name="warning" size={14} color="#ff9800" />
                          <Text style={styles.detailWarningText}>No coin detected - using estimated dimensions</Text>
                        </View>
                        <Text style={styles.detailTip}>
                          💡 Tip: Place a ₱5 coin (25mm) on the LEFT side for accurate sizing
                        </Text>
                      </>
                    )}
                  </View>

                  {/* Dimensions */}
                  <View style={styles.detailCard}>
                    <View style={styles.detailCardHeader}>
                      <Ionicons name="resize" size={14} color="#2196f3" />
                      <Text style={styles.detailCardTitle}>Dimensions</Text>
                    </View>
                    <View style={styles.dimGrid}>
                      <View style={styles.dimGridItem}>
                        <Text style={styles.dimGridLabel}>Length</Text>
                        <Text style={styles.dimGridValue}>{mlResult.dimensions?.length_cm?.toFixed(2) || '—'} cm</Text>
                      </View>
                      <View style={styles.dimGridItem}>
                        <Text style={styles.dimGridLabel}>Width</Text>
                        <Text style={styles.dimGridValue}>{mlResult.dimensions?.width_cm?.toFixed(2) || '—'} cm</Text>
                      </View>
                      <View style={styles.dimGridItem}>
                        <Text style={styles.dimGridLabel}>Kernel Mass</Text>
                        <Text style={styles.dimGridValue}>
                          {mlResult.dimensions?.kernel_mass_g?.toFixed(3) 
                            || mlResult.dimensions?.kernel_weight_g?.toFixed(3) 
                            || '—'} g
                        </Text>
                      </View>
                      <View style={styles.dimGridItem}>
                        <Text style={styles.dimGridLabel}>Fruit Weight</Text>
                        <Text style={styles.dimGridValue}>{mlResult.dimensions?.whole_fruit_weight_g?.toFixed(1) || '—'} g</Text>
                      </View>
                    </View>
                    {mlResult.dimensionsSource && (
                      <Text style={styles.dimSource}>Source: {mlResult.dimensionsSource}</Text>
                    )}
                  </View>

                  {/* Oil Yield Prediction */}
                  <View style={styles.detailCard}>
                    <View style={styles.detailCardHeader}>
                      <Ionicons name="water" size={14} color="#8bc34a" />
                      <Text style={styles.detailCardTitle}>Oil Yield Prediction</Text>
                    </View>
                    <Text style={styles.detailText}>
                      <Text style={styles.detailLabel}>Predicted Yield: </Text>
                      <Text style={[styles.detailValue, styles.detailValueBig]}>{mlResult.oilYieldPercent?.toFixed(1) || '—'}%</Text>
                    </Text>
                    <Text style={styles.detailText}>
                      <Text style={styles.detailLabel}>Category: </Text>
                      <Text style={styles.detailValue}>{mlResult.yieldCategory || 'Unknown'}</Text>
                    </Text>
                    <Text style={styles.detailText}>
                      <Text style={styles.detailLabel}>Confidence: </Text>
                      <Text style={styles.detailValue}>{Math.round((mlResult.oilConfidence || mlResult.overallConfidence || 0) * 100)}%</Text>
                    </Text>
                  </View>

                  {/* Interpretation */}
                  <View style={styles.detailCard}>
                    <View style={styles.detailCardHeader}>
                      <Ionicons name="bulb" size={14} color="#ff9800" />
                      <Text style={styles.detailCardTitle}>Interpretation</Text>
                    </View>
                    <Text style={styles.interpFullText}>{mlResult.interpretation || 'No interpretation available.'}</Text>
                    
                    {/* Recommendation based on category */}
                    <View style={styles.recommendBox}>
                      <Text style={styles.recommendTitle}>📋 Recommendation:</Text>
                      {mlResult.category === 'GREEN' && (
                        <Text style={styles.recommendText}>⏳ This fruit is IMMATURE. Wait for it to turn yellow for higher oil yield.</Text>
                      )}
                      {mlResult.category === 'YELLOW' && (
                        <Text style={styles.recommendText}>✅ This fruit is at OPTIMAL maturity for oil extraction.</Text>
                      )}
                      {mlResult.category === 'BROWN' && (
                        <Text style={styles.recommendText}>⚠️ This fruit is OVERRIPE. Oil quality may be affected.</Text>
                      )}
                    </View>
                  </View>
                </View>
              )}

              {/* Compact view when not expanded */}
              {!showDetails && (
                <>
                  {/* Dimensions Row */}
                  <View style={[styles.dimsRow, isMobile && styles.dimsRowMobile]}>
                    <View style={styles.dimItem}>
                      <Text style={[styles.dimValue, isMobile && styles.dimValueMobile]}>{mlResult.dimensions?.length_cm?.toFixed(1) || '—'}</Text>
                      <Text style={[styles.dimLabel, isMobile && styles.dimLabelMobile]}>L(cm)</Text>
                    </View>
                    <View style={styles.dimItem}>
                      <Text style={[styles.dimValue, isMobile && styles.dimValueMobile]}>{mlResult.dimensions?.width_cm?.toFixed(1) || '—'}</Text>
                      <Text style={[styles.dimLabel, isMobile && styles.dimLabelMobile]}>W(cm)</Text>
                    </View>
                    <View style={styles.dimItem}>
                      <Text style={[styles.dimValue, isMobile && styles.dimValueMobile]}>{mlResult.dimensions?.whole_fruit_weight_g?.toFixed(0) || '—'}</Text>
                      <Text style={[styles.dimLabel, isMobile && styles.dimLabelMobile]}>Wt(g)</Text>
                    </View>
                  </View>

                  {/* Coin Status */}
                  <View style={[styles.coinRow, isMobile && styles.coinRowMobile, { backgroundColor: mlResult.referenceDetected ? '#e8f5e9' : '#fff3e0' }]}>
                    <Ionicons name={mlResult.referenceDetected ? "checkmark-circle" : "warning"} size={isMobile ? 18 : 14} color={mlResult.referenceDetected ? '#4caf50' : '#FFA000'} />
                    <Text style={[styles.coinText, isMobile && styles.coinTextMobile]}>{mlResult.referenceDetected ? 'Coin detected' : 'No coin - estimated'}</Text>
                  </View>

                  {/* Interpretation */}
                  {mlResult.interpretation && (
                    <Text style={[styles.interpText, isMobile && styles.interpTextMobile]} numberOfLines={3}>{mlResult.interpretation}</Text>
                  )}
                </>
              )}

              {/* Actions */}
              <View style={[styles.actionRow, isMobile && styles.actionRowMobile]}>
                <Pressable style={[styles.resetBtn, isMobile && styles.resetBtnMobile]} onPress={resetScan}>
                  <Ionicons name="refresh" size={isMobile ? 18 : 14} color={theme.colors.green} />
                  <Text style={[styles.resetText, isMobile && styles.resetTextMobile]}>New</Text>
                </Pressable>
                <Pressable style={[styles.saveBtn, isMobile && styles.saveBtnMobile, saving && styles.saveBtnDisabled]} onPress={saveResult} disabled={saving}>
                  {saving ? (
                    <ActivityIndicator size="small" color="#fff" />
                  ) : (
                    <>
                      <Ionicons name="save" size={isMobile ? 18 : 14} color="#fff" />
                      <Text style={[styles.saveText, isMobile && styles.saveTextMobile]}>Save</Text>
                    </>
                  )}
                </Pressable>
              </View>
            </ScrollView>
          ) : mlResult && !mlResult.isTalisay ? (
            <View style={[styles.notTalisay, isMobile && styles.notTalisayMobile]}>
              <Ionicons name="close-circle" size={isMobile ? 48 : 32} color="#f44336" />
              <Text style={[styles.notTalisayText, isMobile && styles.notTalisayTextMobile]}>Not Talisay</Text>
              <Pressable style={[styles.tryAgainBtn, isMobile && styles.tryAgainBtnMobile]} onPress={resetScan}>
                <Text style={[styles.tryAgainText, isMobile && styles.tryAgainTextMobile]}>Try Again</Text>
              </Pressable>
            </View>
          ) : (
            <View style={styles.noResult}>
              <Ionicons name="analytics" size={32} color="#ccc" />
              <Text style={styles.noResultText}>Upload an image to see analysis results</Text>
            </View>
          )}
        </View>
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: theme.colors.greenDark,
    padding: 10,
    borderRadius: 8,
    marginBottom: 8,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  headerTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '600',
  },

  mainContent: {
    flex: 1,
    flexDirection: 'row',
    gap: 10,
  },

  leftColumn: {
    flex: 1,
    gap: 6,
  },
  btnRow: {
    flexDirection: 'row',
    gap: 6,
  },
  pickBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    backgroundColor: theme.colors.green,
    paddingVertical: 8,
    borderRadius: 6,
  },
  pickBtnAlt: {
    backgroundColor: '#e8f5e9',
  },
  btnDisabled: {
    backgroundColor: '#f0f0f0',
    opacity: 0.6,
  },
  pickBtnText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },

  imageBox: {
    flex: 1,
    minHeight: 140,
    borderRadius: 8,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#ddd',
    backgroundColor: '#f5f5f5',
  },
  image: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  imageOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.6)',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
  },
  overlayText: {
    color: '#fff',
    fontSize: 11,
  },
  placeholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
  },
  placeholderText: {
    color: '#999',
    fontSize: 11,
  },

  tipBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#fff8e1',
    paddingHorizontal: 8,
    paddingVertical: 6,
    borderRadius: 4,
  },
  tipText: {
    fontSize: 10,
    color: '#5d4037',
  },

  filenameBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#e8f5e9',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  filenameText: {
    fontSize: 9,
    color: theme.colors.greenDark,
    flex: 1,
  },

  errorBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#ffebee',
    padding: 6,
    borderRadius: 4,
  },
  errorText: {
    color: '#f44336',
    fontSize: 10,
    flex: 1,
  },

  rightColumn: {
    flex: 1,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    padding: 10,
    justifyContent: 'center',
  },

  resultMain: {
    alignItems: 'center',
    marginBottom: 8,
  },
  bigPercent: {
    fontSize: 48,
    fontWeight: '900',
    color: theme.colors.greenDark,
    lineHeight: 52,
  },
  resultLabel: {
    fontSize: 11,
    color: '#666',
    marginTop: -4,
  },

  catBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 14,
    alignSelf: 'center',
    marginBottom: 8,
  },
  catText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
  },
  catConf: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 10,
  },

  dimsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: '#fff',
    paddingVertical: 8,
    borderRadius: 6,
    marginBottom: 6,
  },
  dimItem: {
    alignItems: 'center',
  },
  dimValue: {
    fontSize: 16,
    fontWeight: '700',
    color: theme.colors.greenDark,
  },
  dimLabel: {
    fontSize: 9,
    color: '#666',
  },

  coinRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 8,
    paddingVertical: 6,
    borderRadius: 4,
    marginBottom: 6,
  },
  coinText: {
    fontSize: 10,
    color: '#333',
  },

  interpText: {
    fontSize: 10,
    color: '#666',
    lineHeight: 14,
    marginBottom: 8,
  },

  actionRow: {
    flexDirection: 'row',
    gap: 6,
  },
  resetBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    backgroundColor: '#fff',
    paddingVertical: 8,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: theme.colors.green,
  },
  resetText: {
    color: theme.colors.green,
    fontSize: 11,
    fontWeight: '600',
  },
  saveBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    backgroundColor: theme.colors.green,
    paddingVertical: 8,
    borderRadius: 6,
  },
  saveBtnDisabled: {
    opacity: 0.7,
  },
  saveText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '600',
  },

  notTalisay: {
    alignItems: 'center',
    gap: 8,
  },
  notTalisayText: {
    color: '#f44336',
    fontSize: 14,
    fontWeight: '700',
  },
  tryAgainBtn: {
    backgroundColor: '#f44336',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 6,
  },
  tryAgainText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '600',
  },

  noResult: {
    alignItems: 'center',
    gap: 8,
    padding: 16,
  },
  noResultText: {
    color: '#999',
    fontSize: 11,
    textAlign: 'center',
  },

  /* Login Modal Styles */
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    width: '100%',
    maxWidth: 340,
    alignItems: 'center',
  },
  modalHeader: {
    alignItems: 'center',
    marginBottom: 20,
  },
  modalIconWrap: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#e8f5e9',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#333',
    marginBottom: 6,
  },
  modalSubtitle: {
    fontSize: 13,
    color: '#666',
    textAlign: 'center',
    lineHeight: 18,
  },
  modalBtnRow: {
    flexDirection: 'row',
    gap: 10,
    width: '100%',
  },
  modalBtnCancel: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
    alignItems: 'center',
  },
  modalBtnCancelText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#666',
  },
  modalBtnLogin: {
    flex: 1,
    flexDirection: 'row',
    paddingVertical: 12,
    borderRadius: 8,
    backgroundColor: theme.colors.green,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
  },
  modalBtnLoginText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#fff',
  },

  /* Expand Button */
  expandBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    paddingVertical: 6,
    backgroundColor: '#e8f5e9',
    borderRadius: 6,
    marginBottom: 8,
  },
  expandBtnText: {
    fontSize: 10,
    fontWeight: '600',
    color: theme.colors.green,
  },

  /* Expanded Details Section */
  detailsSection: {
    gap: 8,
    marginBottom: 8,
  },
  detailCard: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 10,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  detailCardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 8,
    paddingBottom: 6,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  detailCardTitle: {
    fontSize: 11,
    fontWeight: '700',
    color: '#333',
  },
  detailText: {
    fontSize: 10,
    marginBottom: 3,
  },
  detailLabel: {
    color: '#666',
  },
  detailValue: {
    color: '#333',
    fontWeight: '600',
  },
  detailValueBig: {
    fontSize: 14,
    fontWeight: '800',
  },
  detailSubheader: {
    fontSize: 9,
    fontWeight: '600',
    color: '#666',
    marginTop: 6,
    marginBottom: 4,
  },

  /* Color Probability Bars */
  probRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 3,
  },
  probLabel: {
    width: 50,
    fontSize: 9,
    fontWeight: '600',
  },
  probBarWrap: {
    flex: 1,
    height: 8,
    backgroundColor: '#f0f0f0',
    borderRadius: 4,
    marginHorizontal: 6,
    overflow: 'hidden',
  },
  probBar: {
    height: '100%',
    borderRadius: 4,
  },
  probValue: {
    width: 30,
    fontSize: 9,
    textAlign: 'right',
    color: '#666',
  },

  /* Detail States */
  detailSuccess: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginBottom: 4,
  },
  detailSuccessText: {
    fontSize: 10,
    color: '#4caf50',
    fontWeight: '600',
  },
  detailWarning: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginBottom: 4,
  },
  detailWarningText: {
    fontSize: 10,
    color: '#ff9800',
    fontWeight: '600',
  },
  detailTip: {
    fontSize: 9,
    color: '#666',
    fontStyle: 'italic',
    marginTop: 4,
  },

  /* Dimension Grid */
  dimGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  dimGridItem: {
    width: '50%',
    paddingVertical: 4,
  },
  dimGridLabel: {
    fontSize: 9,
    color: '#666',
  },
  dimGridValue: {
    fontSize: 12,
    fontWeight: '700',
    color: theme.colors.greenDark,
  },
  dimSource: {
    fontSize: 8,
    color: '#999',
    fontStyle: 'italic',
    marginTop: 4,
  },

  /* Spot Detection */
  spotWarning: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#fff3e0',
    padding: 6,
    borderRadius: 4,
    marginTop: 6,
    marginBottom: 2,
  },
  spotWarningText: {
    fontSize: 10,
    color: '#e65100',
    fontWeight: '600',
    flex: 1,
  },
  spotNote: {
    fontSize: 9,
    color: '#666',
    fontStyle: 'italic',
    marginLeft: 18,
    marginBottom: 4,
  },

  /* Interpretation */
  interpFullText: {
    fontSize: 10,
    color: '#333',
    lineHeight: 15,
    marginBottom: 8,
  },
  recommendBox: {
    backgroundColor: '#f5f5f5',
    padding: 8,
    borderRadius: 6,
    marginTop: 4,
  },
  recommendTitle: {
    fontSize: 10,
    fontWeight: '700',
    color: '#333',
    marginBottom: 4,
  },
  recommendText: {
    fontSize: 10,
    color: '#555',
    lineHeight: 14,
  },

  /* ========== MOBILE RESPONSIVE STYLES ========== */
  
  /* Header - Mobile */
  headerMobile: {
    padding: 14,
    borderRadius: 12,
    marginBottom: 12,
  },
  headerTitleMobile: {
    fontSize: 18,
  },

  /* Main Content - Mobile (vertical stacking) */
  mainContentMobile: {
    flexDirection: 'column',
    gap: 12,
  },

  /* Left Column - Mobile */
  leftColumnMobile: {
    gap: 10,
  },

  /* Buttons - Mobile */
  pickBtnMobile: {
    paddingVertical: 14,
    borderRadius: 10,
    gap: 8,
  },
  pickBtnTextMobile: {
    fontSize: 16,
    fontWeight: '700',
  },

  /* Image Box - Mobile */
  imageBoxMobile: {
    minHeight: 200,
    borderRadius: 12,
    borderWidth: 2,
  },
  overlayTextMobile: {
    fontSize: 16,
    fontWeight: '600',
    marginTop: 8,
  },
  placeholderTextMobile: {
    fontSize: 16,
    marginTop: 8,
  },

  /* Filename - Mobile */
  filenameBoxMobile: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
  },
  filenameTextMobile: {
    fontSize: 14,
  },

  /* Tip Box - Mobile */
  tipBoxMobile: {
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderRadius: 8,
    gap: 10,
  },
  tipTextMobile: {
    fontSize: 14,
  },

  /* Error Box - Mobile */
  errorBoxMobile: {
    padding: 12,
    borderRadius: 8,
    gap: 10,
  },
  errorTextMobile: {
    fontSize: 14,
  },

  /* Right Column - Mobile */
  rightColumnMobile: {
    borderRadius: 12,
    padding: 16,
    minHeight: 200,
  },

  /* Result Main - Mobile */
  bigPercentMobile: {
    fontSize: 64,
    lineHeight: 70,
  },
  resultLabelMobile: {
    fontSize: 16,
    marginTop: 0,
  },

  /* Category Badge - Mobile */
  catBadgeMobile: {
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 20,
    marginBottom: 12,
  },
  catTextMobile: {
    fontSize: 16,
  },

  /* Dimensions Row - Mobile */
  dimsRowMobile: {
    paddingVertical: 14,
    borderRadius: 10,
    marginBottom: 10,
  },
  dimValueMobile: {
    fontSize: 22,
    fontWeight: '800',
  },
  dimLabelMobile: {
    fontSize: 12,
    marginTop: 2,
  },

  /* Coin Row - Mobile */
  coinRowMobile: {
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderRadius: 8,
    marginBottom: 10,
    gap: 10,
  },
  coinTextMobile: {
    fontSize: 14,
  },

  /* Interpretation - Mobile */
  interpTextMobile: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 12,
  },

  /* Action Row - Mobile */
  actionRowMobile: {
    gap: 12,
    marginTop: 8,
  },
  resetBtnMobile: {
    paddingVertical: 14,
    borderRadius: 10,
    borderWidth: 2,
    gap: 8,
  },
  resetTextMobile: {
    fontSize: 16,
    fontWeight: '700',
  },
  saveBtnMobile: {
    paddingVertical: 14,
    borderRadius: 10,
    gap: 8,
  },
  saveTextMobile: {
    fontSize: 16,
    fontWeight: '700',
  },

  /* Not Talisay - Mobile */
  notTalisayMobile: {
    padding: 24,
    gap: 16,
  },
  notTalisayTextMobile: {
    fontSize: 20,
  },
  tryAgainBtnMobile: {
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 10,
  },
  tryAgainTextMobile: {
    fontSize: 16,
  },
});
