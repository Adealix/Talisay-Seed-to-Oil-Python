import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Screen from '../components/Screen';
import { Body, Card, Title } from '../components/Ui';
import { theme } from '../theme/theme';

export default function ProposalScreen() {
  return (
    <Screen>
      <Card>
        <Title>System Proposal</Title>
        <Body>
          Machine Learning–Based Prediction of Seed-to-Oil Conversion Ratios in Talisay (Terminalia catappa) Fruits using
          Morphological Feature Analysis
        </Body>
      </Card>

      <Section title="1.1 Introduction">
        Terminalia catappa (Talisay) has seeds containing high-quality oil for culinary, medicinal, and biofuel
        applications. Manual extraction/testing is destructive and labor-intensive. This prototype proposes predicting
        oil conversion ratios by analyzing external morphological features, supporting faster, non-destructive sorting.
      </Section>

      <Section title="1.2 Background of Study">
        Oil potential is limited by seed variability and lack of efficient sorting. Traditional oil determination
        requires crushing seeds. Morphological Feature Analysis (length, width, weight, color) combined with ML can
        learn patterns correlated with oil yield and reduce waste.
      </Section>

      <Section title="1.3 Statement of the Problem">
        • Which physical features are significant predictors of high oil content?
        {'\n'}• Which ML regression algorithm provides the most precise estimation?
        {'\n'}• What is the correlation between external morphology and extracted oil weight?
      </Section>

      <Section title="1.3.1 Main Problem">
        How can Machine Learning be utilized to predict seed-to-oil conversion ratios of Terminalia catappa using only
        external morphological characteristics?
      </Section>

      <Section title="Objectives">
        General: Develop an ML-based model that estimates seed-to-oil conversion ratios using morphology.
        {'\n\n'}Specific:
        {'\n'}• Build a dataset of morphological traits and oil yields.
        {'\n'}• Train/validate a model predicting oil weight from physical dimensions.
        {'\n'}• Evaluate performance using MAE and RMSE.
      </Section>

      <Section title="Significance">
        • Agricultural efficiency: pre-sort fruits without destructive testing.
        {'\n'}• Industrial innovation: faster assessment for biofuel/pharma.
        {'\n'}• Waste reduction: avoid processing low-yield seeds.
      </Section>

      <Section title="Scope and Delimitation">
        Scope: software development + statistical modeling for oil yield prediction from morphology.
        {'\n\n'}Delimitation: excludes chemical refinement and extraction machine engineering; limited to prediction from
        data collected from Philippine samples.
      </Section>

      <Card>
        <Text style={styles.noteTitle}>Prototype note</Text>
        <Body>
          This app currently uses a simple image color heuristic (green/yellow/brown) and optional inputs to produce a
          close-but-not-perfect estimate for demonstration.
        </Body>
      </Card>
    </Screen>
  );
}

function Section({ title, children }) {
  return (
    <Card>
      <View style={{ marginBottom: 8 }}>
        <Text style={styles.sectionTitle}>{title}</Text>
      </View>
      <Body>{children}</Body>
    </Card>
  );
}

const styles = StyleSheet.create({
  sectionTitle: {
    color: theme.colors.greenDark,
    fontSize: 16,
    fontWeight: '800',
  },
  noteTitle: {
    color: theme.colors.muted,
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    marginBottom: 8,
  },
});
