import React from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Screen from '../components/Screen';
import { Body, Card, Kicker, Title, Divider, SectionTitle } from '../components/Ui';
import { theme } from '../theme/theme';

export default function AboutUsScreen() {
  return (
    <Screen>
      {/* Hero Section */}
      <View style={styles.heroSection}>
        <View style={styles.heroIcon}>
          <Ionicons name="leaf" size={48} color="#fff" />
        </View>
        <Title style={styles.heroTitle}>About This Project</Title>
        <Body style={styles.heroSubtitle}>
          Talisay Oil Yield Predictor - A Machine Learning System Proposal
        </Body>
      </View>

      {/* Mission Card */}
      <Card>
        <Kicker>Our Mission</Kicker>
        <Body>
          To develop an accessible, mobile-friendly tool that helps farmers and researchers 
          predict the seed-to-oil conversion ratio of Talisay (Terminalia catappa) fruits 
          using image-based analysis and machine learning techniques.
        </Body>
      </Card>

      {/* About Talisay Card */}
      <Card>
        <Kicker>About Talisay</Kicker>
        <Body>
          Talisay, also known as the Beach Almond or Indian Almond (Terminalia catappa), 
          is a tropical tree widely found in coastal areas of the Philippines. The seeds 
          of the Talisay fruit contain edible oil that has various culinary and potential 
          industrial applications.
        </Body>
        
        <Divider />
        
        <View style={styles.statsRow}>
          <View style={styles.statItem}>
            <Ionicons name="leaf" size={28} color={theme.colors.green} />
            <Body style={styles.statValue}>Green</Body>
            <Body style={styles.statLabel}>Highest Oil Yield</Body>
          </View>
          <View style={styles.statItem}>
            <Ionicons name="sunny" size={28} color="#f5a623" />
            <Body style={styles.statValue}>Yellow</Body>
            <Body style={styles.statLabel}>Medium Oil Yield</Body>
          </View>
          <View style={styles.statItem}>
            <Ionicons name="ellipse" size={28} color="#8b4513" />
            <Body style={styles.statValue}>Brown</Body>
            <Body style={styles.statLabel}>Lower Oil Yield</Body>
          </View>
        </View>
      </Card>

      {/* How It Works Card */}
      <Card>
        <Kicker>How It Works</Kicker>
        <View style={styles.stepList}>
          <View style={styles.stepItem}>
            <View style={styles.stepNumber}>
              <Body style={styles.stepNumberText}>1</Body>
            </View>
            <View style={styles.stepContent}>
              <Body style={styles.stepTitle}>Image Capture</Body>
              <Body style={styles.stepDesc}>
                Take a photo of the Talisay fruit using your device's camera or upload from gallery.
              </Body>
            </View>
          </View>
          
          <View style={styles.stepItem}>
            <View style={styles.stepNumber}>
              <Body style={styles.stepNumberText}>2</Body>
            </View>
            <View style={styles.stepContent}>
              <Body style={styles.stepTitle}>Color Analysis</Body>
              <Body style={styles.stepDesc}>
                The system analyzes the fruit's color to determine its ripeness category.
              </Body>
            </View>
          </View>
          
          <View style={styles.stepItem}>
            <View style={styles.stepNumber}>
              <Body style={styles.stepNumberText}>3</Body>
            </View>
            <View style={styles.stepContent}>
              <Body style={styles.stepTitle}>Ratio Prediction</Body>
              <Body style={styles.stepDesc}>
                Based on the analysis, the ML model predicts the seed-to-oil conversion ratio.
              </Body>
            </View>
          </View>
        </View>
      </Card>

      {/* Team Card */}
      <Card>
        <Kicker>Project Team</Kicker>
        <Body>
          This System Proposal is developed as part of an academic project. The team 
          consists of students researching agricultural technology applications and 
          machine learning for plant analysis.
        </Body>
        
        <Divider />
        
        <View style={styles.contactInfo}>
          <View style={styles.contactRow}>
            <Ionicons name="mail-outline" size={20} color={theme.colors.green} />
            <Body>contact@talisayml.example.com</Body>
          </View>
          <View style={styles.contactRow}>
            <Ionicons name="location-outline" size={20} color={theme.colors.green} />
            <Body>Philippines</Body>
          </View>
        </View>
      </Card>
    </Screen>
  );
}

const styles = StyleSheet.create({
  heroSection: {
    backgroundColor: theme.colors.greenDark,
    borderRadius: 12,
    padding: 32,
    alignItems: 'center',
    marginBottom: 20,
  },
  heroIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  heroTitle: {
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 8,
  },
  heroSubtitle: {
    color: 'rgba(255,255,255,0.8)',
    textAlign: 'center',
  },
  statsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
    marginTop: 8,
  },
  statItem: {
    flex: 1,
    minWidth: 100,
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#f9f9f9',
    borderRadius: 8,
  },
  statValue: {
    fontWeight: '700',
    marginTop: 8,
    textAlign: 'center',
  },
  statLabel: {
    color: theme.colors.muted,
    fontSize: 11,
    textAlign: 'center',
    marginTop: 2,
  },
  stepList: {
    gap: 16,
    marginTop: 8,
  },
  stepItem: {
    flexDirection: 'row',
    gap: 14,
  },
  stepNumber: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: theme.colors.green,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepNumberText: {
    color: '#ffffff',
    fontWeight: '800',
  },
  stepContent: {
    flex: 1,
  },
  stepTitle: {
    fontWeight: '700',
    marginBottom: 4,
  },
  stepDesc: {
    color: theme.colors.muted,
  },
  contactInfo: {
    gap: 12,
  },
  contactRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
});
