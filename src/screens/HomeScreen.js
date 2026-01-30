import React, { memo } from 'react';
import { Pressable, StyleSheet, Text, View, useWindowDimensions } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Screen from '../components/Screen';
import { theme } from '../theme/theme';

const MOBILE_BREAKPOINT = 600;

/* ══════════════════════════════════════════════════════════════════════════
   HOME SCREEN - Compact Layout
   ══════════════════════════════════════════════════════════════════════════ */

const HeroBanner = memo(function HeroBanner({ navigation, isMobile }) {
  return (
    <View style={[styles.heroBanner, isMobile && styles.heroBannerMobile]}>
      <View style={styles.heroOverlay} />
      <View style={[styles.heroContent, isMobile && styles.heroContentMobile]}>
        <Text style={[styles.heroTitle, isMobile && styles.heroTitleMobile]}>TALISAY OIL YIELD</Text>
        <Text style={[styles.heroSubtitle, isMobile && styles.heroSubtitleMobile]}>PREDICTION SYSTEM</Text>
        
        <Text style={[styles.heroTaglineText, isMobile && styles.heroTaglineTextMobile]}>
          ML-Based Seed-to-Oil Ratio Estimation
        </Text>

        <View style={[styles.heroButtons, isMobile && styles.heroButtonsMobile]}>
          <Pressable style={[styles.heroPrimaryBtn, isMobile && styles.heroPrimaryBtnMobile]} onPress={() => navigation.navigate('Scan')}>
            <Ionicons name="scan" size={isMobile ? 22 : 16} color="#fff" />
            <Text style={[styles.heroPrimaryBtnText, isMobile && styles.heroPrimaryBtnTextMobile]}>Start Scanning</Text>
          </Pressable>
          
          <Pressable style={[styles.heroSecondaryBtn, isMobile && styles.heroSecondaryBtnMobile]} onPress={() => navigation.navigate('Proposal')}>
            <Text style={[styles.heroSecondaryBtnText, isMobile && styles.heroSecondaryBtnTextMobile]}>Learn More</Text>
          </Pressable>
        </View>
      </View>
    </View>
  );
});

const QuickActions = memo(function QuickActions({ navigation, isMobile }) {
  return (
    <View style={[styles.quickActions, isMobile && styles.quickActionsMobile]}>
      <Pressable style={[styles.quickActionBtn, isMobile && styles.quickActionBtnMobile, { backgroundColor: '#e8f5e9' }]} onPress={() => navigation.navigate('Scan')}>
        <Ionicons name="camera" size={isMobile ? 28 : 22} color="#2a9d5c" />
        <Text style={[styles.quickActionText, isMobile && styles.quickActionTextMobile]}>Scan</Text>
      </Pressable>
      
      <Pressable style={[styles.quickActionBtn, isMobile && styles.quickActionBtnMobile, { backgroundColor: '#fff3e0' }]} onPress={() => navigation.navigate('History')}>
        <Ionicons name="time" size={isMobile ? 28 : 22} color="#f5a623" />
        <Text style={[styles.quickActionText, isMobile && styles.quickActionTextMobile]}>History</Text>
      </Pressable>
      
      <Pressable style={[styles.quickActionBtn, isMobile && styles.quickActionBtnMobile, { backgroundColor: '#e3f2fd' }]} onPress={() => navigation.navigate('Proposal')}>
        <Ionicons name="document-text" size={isMobile ? 28 : 22} color="#1976d2" />
        <Text style={[styles.quickActionText, isMobile && styles.quickActionTextMobile]}>Proposal</Text>
      </Pressable>
      
      <Pressable style={[styles.quickActionBtn, isMobile && styles.quickActionBtnMobile, { backgroundColor: '#fce4ec' }]} onPress={() => navigation.navigate('Account')}>
        <Ionicons name="person" size={isMobile ? 28 : 22} color="#c2185b" />
        <Text style={[styles.quickActionText, isMobile && styles.quickActionTextMobile]}>Account</Text>
      </Pressable>
    </View>
  );
});

const FeatureCards = memo(function FeatureCards({ isMobile }) {
  return (
    <View style={[styles.infoSection, isMobile && styles.infoSectionMobile]}>
      <View style={[styles.infoCard, isMobile && styles.infoCardMobile]}>
        <View style={[styles.infoIconCircle, isMobile && styles.infoIconCircleMobile]}>
          <Ionicons name="color-palette" size={isMobile ? 24 : 18} color="#fff" />
        </View>
        <View style={styles.infoTextWrap}>
          <Text style={[styles.infoTitle, isMobile && styles.infoTitleMobile]}>Color Analysis</Text>
          <Text style={[styles.infoDesc, isMobile && styles.infoDescMobile]}>Analyzes fruit color to estimate ripeness</Text>
        </View>
      </View>

      <View style={[styles.infoCard, isMobile && styles.infoCardMobile]}>
        <View style={[styles.infoIconCircle, isMobile && styles.infoIconCircleMobile, { backgroundColor: '#f5a623' }]}>
          <Ionicons name="calculator" size={isMobile ? 24 : 18} color="#fff" />
        </View>
        <View style={styles.infoTextWrap}>
          <Text style={[styles.infoTitle, isMobile && styles.infoTitleMobile]}>Ratio Prediction</Text>
          <Text style={[styles.infoDesc, isMobile && styles.infoDescMobile]}>Predicts seed-to-oil ratio using ML</Text>
        </View>
      </View>

      <View style={[styles.infoCard, isMobile && styles.infoCardMobile]}>
        <View style={[styles.infoIconCircle, isMobile && styles.infoIconCircleMobile, { backgroundColor: '#1976d2' }]}>
          <Ionicons name="trending-up" size={isMobile ? 24 : 18} color="#fff" />
        </View>
        <View style={styles.infoTextWrap}>
          <Text style={[styles.infoTitle, isMobile && styles.infoTitleMobile]}>Expected Trend</Text>
          <Text style={[styles.infoDesc, isMobile && styles.infoDescMobile]}>Green → Yellow → Brown maturity</Text>
        </View>
      </View>
    </View>
  );
});

function HomeScreen({ navigation }) {
  const { width: screenWidth } = useWindowDimensions();
  const isMobile = screenWidth < MOBILE_BREAKPOINT;

  return (
    <Screen scroll={false}>
      <HeroBanner navigation={navigation} isMobile={isMobile} />
      <QuickActions navigation={navigation} isMobile={isMobile} />
      <FeatureCards isMobile={isMobile} />
    </Screen>
  );
}

const styles = StyleSheet.create({
  heroBanner: {
    backgroundColor: theme.colors.greenDark,
    borderRadius: 10,
    overflow: 'hidden',
    marginBottom: 12,
    position: 'relative',
  },
  heroOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,102,51,0.85)',
  },
  heroContent: {
    padding: 20,
    alignItems: 'center',
  },
  heroTitle: {
    color: '#ffffff',
    fontSize: 28,
    fontWeight: '900',
    textAlign: 'center',
    letterSpacing: 1,
  },
  heroSubtitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '300',
    textAlign: 'center',
    marginTop: 2,
  },
  heroTaglineText: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 11,
    marginTop: 8,
  },
  heroButtons: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 16,
  },
  heroPrimaryBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: theme.colors.green,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 6,
  },
  heroPrimaryBtnText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '700',
  },
  heroSecondaryBtn: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#ffffff',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 6,
  },
  heroSecondaryBtnText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '600',
  },

  quickActions: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 12,
  },
  quickActionBtn: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
    gap: 4,
  },
  quickActionText: {
    color: theme.colors.text,
    fontSize: 10,
    fontWeight: '700',
  },

  infoSection: {
    gap: 8,
    flex: 1,
  },
  infoCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: '#ffffff',
    borderRadius: 8,
    padding: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  infoIconCircle: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: theme.colors.green,
    alignItems: 'center',
    justifyContent: 'center',
  },
  infoTextWrap: {
    flex: 1,
  },
  infoTitle: {
    color: theme.colors.text,
    fontSize: 13,
    fontWeight: '700',
  },
  infoDesc: {
    color: theme.colors.muted,
    fontSize: 11,
    marginTop: 2,
  },

  /* ========== MOBILE RESPONSIVE STYLES ========== */

  /* Hero Banner - Mobile */
  heroBannerMobile: {
    borderRadius: 16,
    marginBottom: 16,
  },
  heroContentMobile: {
    padding: 28,
  },
  heroTitleMobile: {
    fontSize: 32,
    letterSpacing: 2,
  },
  heroSubtitleMobile: {
    fontSize: 20,
    marginTop: 4,
  },
  heroTaglineTextMobile: {
    fontSize: 14,
    marginTop: 12,
  },
  heroButtonsMobile: {
    marginTop: 24,
    gap: 14,
  },
  heroPrimaryBtnMobile: {
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 10,
    gap: 10,
  },
  heroPrimaryBtnTextMobile: {
    fontSize: 16,
    fontWeight: '800',
  },
  heroSecondaryBtnMobile: {
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 10,
    borderWidth: 2,
  },
  heroSecondaryBtnTextMobile: {
    fontSize: 16,
    fontWeight: '700',
  },

  /* Quick Actions - Mobile */
  quickActionsMobile: {
    gap: 12,
    marginBottom: 16,
  },
  quickActionBtnMobile: {
    paddingVertical: 18,
    borderRadius: 12,
    gap: 8,
  },
  quickActionTextMobile: {
    fontSize: 14,
    fontWeight: '800',
  },

  /* Info Section - Mobile */
  infoSectionMobile: {
    gap: 12,
  },
  infoCardMobile: {
    gap: 16,
    padding: 16,
    borderRadius: 12,
  },
  infoIconCircleMobile: {
    width: 48,
    height: 48,
    borderRadius: 24,
  },
  infoTitleMobile: {
    fontSize: 16,
    fontWeight: '800',
  },
  infoDescMobile: {
    fontSize: 14,
    marginTop: 4,
    lineHeight: 20,
  },
});

export default memo(HomeScreen);
