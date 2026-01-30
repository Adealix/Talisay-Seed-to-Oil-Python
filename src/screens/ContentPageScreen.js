import React, { memo } from 'react';
import { StyleSheet, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Screen from '../components/Screen';
import { Body, Card, Kicker, Title, Divider } from '../components/Ui';
import { theme } from '../theme/theme';

const ContentPageScreen = memo(function ContentPageScreen({ title, subtitle, icon = 'document-text' }) {
  return (
    <Screen>
      {/* Page Header */}
      <View style={styles.pageHeader}>
        <View style={styles.headerIcon}>
          <Ionicons name={icon} size={32} color="#fff" />
        </View>
        <View>
          <Title style={styles.pageTitle}>{title}</Title>
          <Body style={styles.pageSubtitle}>{subtitle}</Body>
        </View>
      </View>

      {/* Main Content Card */}
      <Card>
        <Kicker>Overview</Kicker>
        <Body>
          This section provides information about {title.toLowerCase()}. 
          As this is a prototype, content will be added as the project develops.
        </Body>
        
        <Divider />
        
        <Kicker>Features</Kicker>
        <View style={styles.featureList}>
          <View style={styles.featureItem}>
            <View style={styles.featureBullet} />
            <Body>Responsive design for mobile and web</Body>
          </View>
          <View style={styles.featureItem}>
            <View style={styles.featureBullet} />
            <Body>Image-based fruit analysis</Body>
          </View>
          <View style={styles.featureItem}>
            <View style={styles.featureBullet} />
            <Body>Seed-to-oil ratio prediction</Body>
          </View>
          <View style={styles.featureItem}>
            <View style={styles.featureBullet} />
            <Body>History tracking and data storage</Body>
          </View>
        </View>
      </Card>

      {/* Info Card */}
      <Card style={styles.infoCard}>
        <View style={styles.infoRow}>
          <Ionicons name="information-circle" size={24} color={theme.colors.green} />
          <Body style={styles.infoText}>
            This is a school System Proposal prototype. Full content and functionality 
            will be implemented in the production version.
          </Body>
        </View>
      </Card>
    </Screen>
  );
});

const styles = StyleSheet.create({
  pageHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    marginBottom: 20,
    padding: 20,
    backgroundColor: theme.colors.greenDark,
    borderRadius: 10,
  },
  headerIcon: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  pageTitle: {
    color: '#ffffff',
    marginBottom: 0,
  },
  pageSubtitle: {
    color: 'rgba(255,255,255,0.8)',
    marginTop: 4,
  },
  featureList: {
    gap: 10,
    marginTop: 8,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  featureBullet: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: theme.colors.green,
  },
  infoCard: {
    backgroundColor: '#e8f5e9',
    borderColor: theme.colors.green,
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
  },
  infoText: {
    flex: 1,
    color: theme.colors.greenDark,
  },
});

export default ContentPageScreen;
