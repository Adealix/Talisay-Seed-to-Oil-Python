import React, { useMemo, useState } from 'react';
import { Image, Platform, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import Screen from '../components/Screen';
import { Body, Button, Card, Label, Subtitle, Title, Kicker, Divider } from '../components/Ui';
import { analyzeFruitImage, predictRatio } from '../utils';
import { addHistoryItem } from '../storage';
import { predictionService } from '../services';
import { parseNumber, getCategoryLabel } from '../utils';
import { theme } from '../theme/theme';

export default function ScanScreen({ navigation }) {
  const [imageUri, setImageUri] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);

  const [lengthText, setLengthText] = useState('');
  const [widthText, setWidthText] = useState('');
  const [weightText, setWeightText] = useState('');

  const lengthMm = useMemo(() => parseNumber(lengthText), [lengthText]);
  const widthMm = useMemo(() => parseNumber(widthText), [widthText]);
  const weightG = useMemo(() => parseNumber(weightText), [weightText]);

  const category = analysis?.category ?? 'BROWN';
  const ratio = useMemo(() => predictRatio({ category, lengthMm, widthMm, weightG }), [category, lengthMm, widthMm, weightG]);

  async function pickImage(fromCamera) {
    setError(null);

    if (fromCamera) {
      const perm = await ImagePicker.requestCameraPermissionsAsync();
      if (!perm.granted) {
        setError('Camera permission is required.');
        return;
      }
    } else {
      const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!perm.granted) {
        setError('Media library permission is required.');
        return;
      }
    }

    const result = fromCamera
      ? await ImagePicker.launchCameraAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          quality: 0.9,
        })
      : await ImagePicker.launchImageLibraryAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          quality: 0.9,
        });

    if (result.canceled) return;
    const uri = result.assets?.[0]?.uri;
    if (!uri) return;

    setImageUri(uri);
    setAnalysis(null);

    setAnalyzing(true);
    try {
      const a = await analyzeFruitImage(uri);
      setAnalysis(a);
    } catch (e) {
      console.warn('[ScanScreen.pickImage]', e?.message);
      setError(
        Platform.OS === 'web'
          ? 'Image analysis failed in web mode. You can still select the category manually.'
          : 'Image analysis failed. You can still select the category manually.'
      );
    } finally {
      setAnalyzing(false);
    }
  }

  async function saveResult() {
    const item = {
      id: String(Date.now()),
      createdAt: new Date().toISOString(),
      imageUri,
      category,
      confidence: analysis?.confidence ?? null,
      ratio,
      inputs: {
        lengthMm: lengthMm ?? null,
        widthMm: widthMm ?? null,
        weightG: weightG ?? null,
      },
    };

    // Best-effort backend save (prototype): if server is not running, local save still works.
    await predictionService.createPrediction({
      category: item.category,
      confidence: item.confidence,
      ratio: item.ratio,
      inputs: item.inputs,
      imageUri: item.imageUri,
    });

    await addHistoryItem(item);
    navigation.navigate('History');
  }

  return (
    <Screen>
      {/* Header Section */}
      <View style={styles.pageHeader}>
        <View style={styles.headerIcon}>
          <Ionicons name="scan" size={32} color="#fff" />
        </View>
        <View style={styles.headerText}>
          <Title style={styles.headerTitle}>Scan Fruit</Title>
          <Body style={styles.headerSubtitle}>Analyze Talisay fruit and predict oil yield</Body>
        </View>
      </View>

      {/* Image Capture Card */}
      <Card>
        <Kicker>Step 1: Capture Image</Kicker>
        <Body style={{ marginBottom: 16 }}>
          Take a photo or select an image of a Talisay fruit for analysis.
        </Body>

        <View style={styles.buttonRow}>
          <Pressable style={styles.actionBtn} onPress={() => pickImage(false)}>
            <View style={styles.actionBtnIcon}>
              <Ionicons name="images" size={28} color="#fff" />
            </View>
            <Text style={styles.actionBtnText}>Pick from Gallery</Text>
          </Pressable>
          
          <Pressable style={[styles.actionBtn, styles.actionBtnSecondary]} onPress={() => pickImage(true)}>
            <View style={[styles.actionBtnIcon, styles.actionBtnIconSecondary]}>
              <Ionicons name="camera" size={28} color={theme.colors.green} />
            </View>
            <Text style={[styles.actionBtnText, styles.actionBtnTextSecondary]}>Use Camera</Text>
          </Pressable>
        </View>

        {error ? (
          <View style={styles.errorBox}>
            <Ionicons name="alert-circle" size={20} color={theme.colors.danger} />
            <Text style={styles.errorText}>{error}</Text>
          </View>
        ) : null}

        {imageUri ? (
          <View style={styles.previewWrap}>
            <Image source={{ uri: imageUri }} style={styles.preview} />
            <View style={styles.previewOverlay}>
              <Ionicons name="checkmark-circle" size={24} color="#fff" />
              <Text style={styles.previewLabel}>Image loaded</Text>
            </View>
          </View>
        ) : (
          <View style={styles.placeholderWrap}>
            <Ionicons name="image-outline" size={48} color="#ccc" />
            <Text style={styles.placeholderText}>No image selected</Text>
          </View>
        )}
      </Card>

      {/* Analysis Card */}
      <Card>
        <Kicker>Step 2: Fruit Category</Kicker>
        <Body style={{ marginBottom: 12 }}>
          {analyzing ? 'Analyzing image...' : 'Select or confirm the detected category:'}
        </Body>

        <View style={styles.categoryRow}>
          <CategoryButton
            label="Green"
            icon="leaf"
            color="#4caf50"
            selected={category === 'GREEN'}
            onPress={() => setAnalysis((prev) => ({ ...(prev ?? {}), category: 'GREEN', confidence: prev?.confidence ?? null }))}
          />
          <CategoryButton
            label="Yellow"
            icon="sunny"
            color="#ffc107"
            selected={category === 'YELLOW'}
            onPress={() => setAnalysis((prev) => ({ ...(prev ?? {}), category: 'YELLOW', confidence: prev?.confidence ?? null }))}
          />
          <CategoryButton
            label="Brown"
            icon="ellipse"
            color="#8b4513"
            selected={category === 'BROWN'}
            onPress={() => setAnalysis((prev) => ({ ...(prev ?? {}), category: 'BROWN', confidence: prev?.confidence ?? null }))}
          />
        </View>

        {analysis?.confidence != null && (
          <View style={styles.confidenceBox}>
            <Ionicons name="analytics" size={18} color={theme.colors.green} />
            <Text style={styles.confidenceText}>
              Confidence: ~{Math.round(analysis.confidence * 100)}%
            </Text>
          </View>
        )}

        <Divider />

        <Kicker>Optional: Morphology Inputs</Kicker>
        <View style={styles.inputsGrid}>
          <MorphField label="Length" unit="mm" value={lengthText} onChangeText={setLengthText} icon="resize" />
          <MorphField label="Width" unit="mm" value={widthText} onChangeText={setWidthText} icon="swap-horizontal" />
          <MorphField label="Weight" unit="g" value={weightText} onChangeText={setWeightText} icon="scale" />
        </View>
      </Card>

      {/* Results Card */}
      <Card style={styles.resultsCard}>
        <View style={styles.resultsHeader}>
          <Ionicons name="analytics" size={24} color="#fff" />
          <Text style={styles.resultsLabel}>PREDICTED RATIO</Text>
        </View>
        
        <Text style={styles.bigNumber}>{Math.round(ratio * 100)}%</Text>
        <Text style={styles.ratioHint}>Seed-to-Oil Conversion Ratio (Estimate)</Text>
        
        <View style={styles.trendBox}>
          <Ionicons name="trending-up" size={18} color={theme.colors.greenDark} />
          <Text style={styles.trendText}>Expected: Green &gt; Yellow &gt; Brown</Text>
        </View>

        <Pressable style={styles.saveBtn} onPress={saveResult} disabled={!imageUri && !analysis}>
          <Ionicons name="save" size={20} color="#fff" />
          <Text style={styles.saveBtnText}>Save to History</Text>
        </Pressable>
      </Card>
    </Screen>
  );
}

/* ──────────────────────────────────────────────────────────────────────────
   Sub-Components
   ────────────────────────────────────────────────────────────────────────── */
function CategoryButton({ label, icon, color, selected, onPress }) {
  return (
    <Pressable
      style={[
        styles.categoryBtn,
        selected && { backgroundColor: color, borderColor: color },
      ]}
      onPress={onPress}
    >
      <Ionicons name={icon} size={24} color={selected ? '#fff' : color} />
      <Text style={[styles.categoryLabel, selected && { color: '#fff' }]}>{label}</Text>
    </Pressable>
  );
}

function MorphField({ label, unit, value, onChangeText, icon }) {
  return (
    <View style={styles.morphField}>
      <View style={styles.morphIcon}>
        <Ionicons name={icon} size={18} color={theme.colors.green} />
      </View>
      <View style={styles.morphContent}>
        <Text style={styles.morphLabel}>{label} ({unit})</Text>
        <TextInput
          value={value}
          onChangeText={onChangeText}
          placeholder="0"
          placeholderTextColor="#999"
          keyboardType={Platform.OS === 'web' ? 'default' : 'numeric'}
          style={styles.morphInput}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  /* Page Header */
  pageHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    marginBottom: 20,
    padding: 20,
    backgroundColor: theme.colors.greenDark,
    borderRadius: 12,
  },
  headerIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerText: {
    flex: 1,
  },
  headerTitle: {
    color: '#ffffff',
    marginBottom: 0,
  },
  headerSubtitle: {
    color: 'rgba(255,255,255,0.8)',
    marginTop: 4,
  },

  /* Button Row */
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  actionBtn: {
    flex: 1,
    backgroundColor: theme.colors.green,
    borderRadius: 10,
    padding: 16,
    alignItems: 'center',
    gap: 8,
  },
  actionBtnSecondary: {
    backgroundColor: '#e8f5e9',
  },
  actionBtnIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionBtnIconSecondary: {
    backgroundColor: 'rgba(42,157,92,0.15)',
  },
  actionBtnText: {
    color: '#ffffff',
    fontWeight: '700',
    fontSize: 13,
  },
  actionBtnTextSecondary: {
    color: theme.colors.greenDark,
  },

  /* Error Box */
  errorBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: '#ffebee',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  errorText: {
    color: theme.colors.danger,
    flex: 1,
  },

  /* Preview */
  previewWrap: {
    borderRadius: 12,
    overflow: 'hidden',
    position: 'relative',
    borderWidth: 2,
    borderColor: theme.colors.green,
  },
  preview: {
    width: '100%',
    height: 200,
    resizeMode: 'cover',
  },
  previewOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(42,157,92,0.9)',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 10,
  },
  previewLabel: {
    color: '#ffffff',
    fontWeight: '700',
  },
  placeholderWrap: {
    height: 150,
    backgroundColor: '#f5f5f5',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#e0e0e0',
    borderStyle: 'dashed',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  placeholderText: {
    color: '#999',
    fontSize: 13,
  },

  /* Category Buttons */
  categoryRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 12,
  },
  categoryBtn: {
    flex: 1,
    padding: 16,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#e0e0e0',
    backgroundColor: '#fff',
    alignItems: 'center',
    gap: 8,
  },
  categoryLabel: {
    fontWeight: '700',
    fontSize: 13,
    color: theme.colors.text,
  },

  /* Confidence */
  confidenceBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#e8f5e9',
    padding: 10,
    borderRadius: 8,
  },
  confidenceText: {
    color: theme.colors.greenDark,
    fontWeight: '600',
    fontSize: 13,
  },

  /* Morphology Inputs */
  inputsGrid: {
    gap: 10,
    marginTop: 8,
  },
  morphField: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f9f9f9',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    overflow: 'hidden',
  },
  morphIcon: {
    width: 48,
    height: 48,
    backgroundColor: '#e8f5e9',
    alignItems: 'center',
    justifyContent: 'center',
  },
  morphContent: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
  },
  morphLabel: {
    flex: 1,
    color: theme.colors.muted,
    fontSize: 13,
    fontWeight: '600',
  },
  morphInput: {
    width: 80,
    paddingVertical: 12,
    paddingHorizontal: 12,
    fontSize: 16,
    fontWeight: '700',
    textAlign: 'right',
    color: theme.colors.text,
  },

  /* Results Card */
  resultsCard: {
    backgroundColor: '#e8f5e9',
    borderColor: theme.colors.green,
    alignItems: 'center',
  },
  resultsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: theme.colors.green,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginBottom: 16,
  },
  resultsLabel: {
    color: '#ffffff',
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  bigNumber: {
    color: theme.colors.greenDark,
    fontSize: 64,
    fontWeight: '900',
  },
  ratioHint: {
    color: theme.colors.muted,
    fontSize: 13,
    marginTop: 4,
    marginBottom: 16,
  },
  trendBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#fff',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
    marginBottom: 20,
  },
  trendText: {
    color: theme.colors.greenDark,
    fontWeight: '600',
    fontSize: 13,
  },
  saveBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: theme.colors.green,
    paddingHorizontal: 28,
    paddingVertical: 14,
    borderRadius: 10,
  },
  saveBtnText: {
    color: '#ffffff',
    fontWeight: '800',
    fontSize: 15,
  },
});
