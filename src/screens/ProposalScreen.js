import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Screen from '../components/Screen';
import { theme } from '../theme/theme';

export default function ProposalScreen() {
  return (
    <Screen>
      {/* Compact Header */}
      <View style={styles.header}>
        <Ionicons name="document-text" size={20} color="#fff" />
        <Text style={styles.headerTitle}>System Proposal</Text>
      </View>

      {/* Title Card */}
      <View style={styles.titleCard}>
        <Text style={styles.mainTitle}>ML-Based Prediction of Seed-to-Oil Conversion Ratios</Text>
        <Text style={styles.subtitle}>Talisay (Terminalia catappa) using Morphological Feature Analysis</Text>
      </View>

      {/* Sections */}
      <Section title="Introduction">
        Terminalia catappa (Talisay) seeds contain high-quality oil for culinary, medicinal, and biofuel applications. Manual extraction is destructive and labor-intensive. This prototype predicts oil conversion ratios by analyzing external morphological features.
      </Section>

      <Section title="Background">
        Oil potential is limited by seed variability. Traditional oil determination requires crushing seeds. Morphological Feature Analysis (length, width, weight, color) combined with ML can learn patterns correlated with oil yield.
      </Section>

      <Section title="Problem Statement">
        • Which physical features are significant predictors of high oil content?{'\n'}
        • Which ML regression algorithm provides the most precise estimation?{'\n'}
        • What is the correlation between external morphology and extracted oil weight?
      </Section>

      <Section title="Main Problem">
        How can Machine Learning predict seed-to-oil conversion ratios of Terminalia catappa using only external morphological characteristics?
      </Section>

      <Section title="Objectives">
        General: Develop an ML model that estimates seed-to-oil ratios using morphology.{'\n\n'}
        Specific:{'\n'}
        • Build a dataset of morphological traits and oil yields{'\n'}
        • Train/validate a model predicting oil weight from physical dimensions{'\n'}
        • Evaluate performance using MAE and RMSE
      </Section>

      <Section title="Significance">
        • Agricultural efficiency: pre-sort fruits without destructive testing{'\n'}
        • Industrial innovation: faster assessment for biofuel/pharma{'\n'}
        • Waste reduction: avoid processing low-yield seeds
      </Section>

      <Section title="Scope">
        Software development + statistical modeling for oil yield prediction from morphology. Excludes chemical refinement and extraction machine engineering.
      </Section>

      <View style={styles.noteBox}>
        <Ionicons name="information-circle" size={16} color="#666" />
        <Text style={styles.noteText}>This app uses image color heuristics (green/yellow/brown) and ML to produce oil yield estimates.</Text>
      </View>
    </Screen>
  );
}

function Section({ title, children }) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      <Text style={styles.sectionBody}>{children}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: theme.colors.greenDark,
    padding: 10,
    borderRadius: 8,
    marginBottom: 10,
  },
  headerTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },

  titleCard: {
    backgroundColor: '#e8f5e9',
    padding: 12,
    borderRadius: 8,
    marginBottom: 10,
  },
  mainTitle: {
    fontSize: 14,
    fontWeight: '800',
    color: theme.colors.greenDark,
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 11,
    color: '#666',
  },

  section: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 10,
    marginBottom: 8,
  },
  sectionTitle: {
    fontSize: 12,
    fontWeight: '800',
    color: theme.colors.greenDark,
    marginBottom: 4,
  },
  sectionBody: {
    fontSize: 11,
    color: '#333',
    lineHeight: 16,
  },

  noteBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    backgroundColor: '#f5f5f5',
    padding: 10,
    borderRadius: 8,
    marginBottom: 16,
  },
  noteText: {
    flex: 1,
    fontSize: 10,
    color: '#666',
    lineHeight: 14,
  },
});
