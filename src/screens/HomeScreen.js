import React, { memo, useMemo } from 'react';
import { Image, Pressable, ScrollView, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Screen from '../components/Screen';
import { theme } from '../theme/theme';

/* ══════════════════════════════════════════════════════════════════════════
   HOME SCREEN - UCAP-Style Layout with Hero, News, Events, Publications
   ══════════════════════════════════════════════════════════════════════════ */

/* ──────────────────────────────────────────────────────────────────────────
   Hero Banner Section - Like UCAP's "FUTURE-PROOFING THE COCONUT"
   ────────────────────────────────────────────────────────────────────────── */
const HeroBanner = memo(function HeroBanner({ navigation }) {
  return (
    <View style={styles.heroBanner}>
      {/* Background Gradient Overlay */}
      <View style={styles.heroOverlay} />
      
      {/* Hero Content */}
      <View style={styles.heroContent}>
        <Text style={styles.heroTitle}>TALISAY OIL YIELD</Text>
        <Text style={styles.heroSubtitle}>PREDICTION SYSTEM</Text>
        
        <View style={styles.heroTagline}>
          <Text style={styles.heroTaglineText}>
            Machine Learning-Based Seed-to-Oil Ratio Estimation
          </Text>
        </View>

        {/* Action Buttons */}
        <View style={styles.heroButtons}>
          <Pressable 
            style={styles.heroPrimaryBtn}
            onPress={() => navigation.navigate('Scan')}
          >
            <Ionicons name="scan" size={20} color="#fff" />
            <Text style={styles.heroPrimaryBtnText}>Start Scanning</Text>
          </Pressable>
          
          <Pressable 
            style={styles.heroSecondaryBtn}
            onPress={() => navigation.navigate('Proposal')}
          >
            <Text style={styles.heroSecondaryBtnText}>Learn More</Text>
          </Pressable>
        </View>
      </View>

      {/* Social Links - Like UCAP */}
      <View style={styles.heroSocialRow}>
        <Pressable style={styles.heroSocialBtn}>
          <Ionicons name="logo-facebook" size={18} color="#fff" />
        </Pressable>
        <Pressable style={styles.heroSocialBtn}>
          <Ionicons name="logo-twitter" size={18} color="#fff" />
        </Pressable>
      </View>
    </View>
  );
});

/* ──────────────────────────────────────────────────────────────────────────
   Section Header - "TOP NEWS" style
   ────────────────────────────────────────────────────────────────────────── */
const SectionHeader = memo(function SectionHeader({ title, showArrows = true }) {
  return (
    <View style={styles.sectionHeader}>
      <View style={styles.sectionTitleRow}>
        <View style={styles.sectionTitleLine} />
        <Text style={styles.sectionTitle}>{title}</Text>
        <View style={styles.sectionTitleLine} />
      </View>
      
      {showArrows && (
        <View style={styles.sectionArrows}>
          <Pressable style={styles.arrowBtn}>
            <Ionicons name="chevron-back" size={20} color="#666" />
          </Pressable>
          <Pressable style={styles.arrowBtn}>
            <Ionicons name="chevron-forward" size={20} color="#666" />
          </Pressable>
        </View>
      )}
    </View>
  );
});

/* ──────────────────────────────────────────────────────────────────────────
   News Card - Like UCAP's news grid
   ────────────────────────────────────────────────────────────────────────── */
const NewsCard = memo(function NewsCard({ title, summary, date, onPress }) {
  return (
    <Pressable style={styles.newsCard} onPress={onPress}>
      <View style={styles.newsImagePlaceholder}>
        <Ionicons name="newspaper-outline" size={40} color="#ccc" />
      </View>
      <View style={styles.newsContent}>
        <Text style={styles.newsTitle} numberOfLines={2}>{title}</Text>
        <Text style={styles.newsSummary} numberOfLines={3}>{summary}</Text>
        <Text style={styles.newsReadMore}>Read more &gt;</Text>
      </View>
    </Pressable>
  );
});

/* ──────────────────────────────────────────────────────────────────────────
   Event Card - Like UCAP's TOP EVENTS
   ────────────────────────────────────────────────────────────────────────── */
const EventCard = memo(function EventCard({ icon, title, description, onPress }) {
  return (
    <Pressable style={styles.eventCard} onPress={onPress}>
      <View style={styles.eventImageArea}>
        <Ionicons name={icon} size={48} color="#2a9d5c" />
      </View>
      <Text style={styles.eventTitle}>{title}</Text>
      <Text style={styles.eventDesc} numberOfLines={3}>{description}</Text>
      <Text style={styles.eventReadMore}>Read more &gt;</Text>
    </Pressable>
  );
});

/* ──────────────────────────────────────────────────────────────────────────
   Publication Card - Like UCAP PUBLICATIONS
   ────────────────────────────────────────────────────────────────────────── */
const PublicationCard = memo(function PublicationCard({ icon, title, subtitle }) {
  return (
    <Pressable style={styles.pubCard}>
      <View style={styles.pubIcon}>
        <Ionicons name={icon} size={36} color="#2a9d5c" />
      </View>
      <Text style={styles.pubTitle}>{title}</Text>
      <Text style={styles.pubSubtitle}>{subtitle}</Text>
    </Pressable>
  );
});

/* ──────────────────────────────────────────────────────────────────────────
   Quick Action Button Row - Like UCAP sidebar images
   ────────────────────────────────────────────────────────────────────────── */
const QuickActions = memo(function QuickActions({ navigation }) {
  return (
    <View style={styles.quickActions}>
      <Pressable 
        style={[styles.quickActionBtn, { backgroundColor: '#e8f5e9' }]}
        onPress={() => navigation.navigate('Scan')}
      >
        <Ionicons name="camera" size={32} color="#2a9d5c" />
        <Text style={styles.quickActionText}>Scan Fruit</Text>
      </Pressable>
      
      <Pressable 
        style={[styles.quickActionBtn, { backgroundColor: '#fff3e0' }]}
        onPress={() => navigation.navigate('History')}
      >
        <Ionicons name="time" size={32} color="#f5a623" />
        <Text style={styles.quickActionText}>View History</Text>
      </Pressable>
      
      <Pressable 
        style={[styles.quickActionBtn, { backgroundColor: '#e3f2fd' }]}
        onPress={() => navigation.navigate('Proposal')}
      >
        <Ionicons name="document-text" size={32} color="#1976d2" />
        <Text style={styles.quickActionText}>Proposal</Text>
      </Pressable>
      
      <Pressable 
        style={[styles.quickActionBtn, { backgroundColor: '#fce4ec' }]}
        onPress={() => navigation.navigate('Account')}
      >
        <Ionicons name="person" size={32} color="#c2185b" />
        <Text style={styles.quickActionText}>Account</Text>
      </Pressable>
    </View>
  );
});

/* ══════════════════════════════════════════════════════════════════════════
   MAIN HOME SCREEN COMPONENT
   ══════════════════════════════════════════════════════════════════════════ */
function HomeScreen({ navigation }) {
  const { width } = useWindowDimensions();
  const isDesktop = useMemo(() => width >= 1024, [width]);
  const isTablet = useMemo(() => width >= 768, [width]);

  // Sample news data - memoized to prevent recreation
  const newsItems = useMemo(() => [
    {
      title: 'Understanding Talisay Fruit Oil Extraction',
      summary: 'Learn about the process of extracting oil from Talisay fruits and how maturity affects yield.',
      date: 'January 2026',
    },
    {
      title: 'Machine Learning in Agriculture',
      summary: 'How ML models can predict crop yields and optimize agricultural processes.',
      date: 'January 2026',
    },
    {
      title: 'Seed-to-Oil Ratio Research',
      summary: 'New studies show correlation between fruit color and oil content in tropical beach almonds.',
      date: 'December 2025',
    },
    {
      title: 'Mobile Apps for Farm Analysis',
      summary: 'The rise of smartphone-based agricultural tools for small-scale farmers.',
      date: 'December 2025',
    },
  ], []);

  return (
    <Screen>
      {/* Hero Banner */}
      <HeroBanner navigation={navigation} />

      {/* Quick Actions (Mobile/Tablet) */}
      {!isDesktop && <QuickActions navigation={navigation} />}

      {/* ═══════════════════════════════════════════════════════════════════
          TOP NEWS Section
          ═══════════════════════════════════════════════════════════════════ */}
      <SectionHeader title="TOP NEWS" />
      
      <ScrollView 
        horizontal 
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.newsScroll}
      >
        {newsItems.map((item, index) => (
          <NewsCard
            key={index}
            title={item.title}
            summary={item.summary}
            date={item.date}
            onPress={() => navigation.navigate('News')}
          />
        ))}
      </ScrollView>

      {/* ═══════════════════════════════════════════════════════════════════
          TOP EVENTS Section
          ═══════════════════════════════════════════════════════════════════ */}
      <SectionHeader title="TOP EVENTS" />
      
      <ScrollView 
        horizontal 
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.eventsScroll}
      >
        <EventCard
          icon="flask"
          title="Research Demo"
          description="Live demonstration of the Talisay oil extraction prediction system using image analysis."
          onPress={() => navigation.navigate('Events')}
        />
        <EventCard
          icon="school"
          title="System Presentation"
          description="School presentation of the ML-based oil yield prediction prototype."
          onPress={() => navigation.navigate('Events')}
        />
        <EventCard
          icon="people"
          title="Community Outreach"
          description="Sharing knowledge about Talisay fruit utilization with local communities."
          onPress={() => navigation.navigate('Events')}
        />
        <EventCard
          icon="trophy"
          title="Competition Entry"
          description="Submission to agricultural technology innovation competitions."
          onPress={() => navigation.navigate('Events')}
        />
      </ScrollView>

      {/* ═══════════════════════════════════════════════════════════════════
          PUBLICATIONS Section
          ═══════════════════════════════════════════════════════════════════ */}
      <SectionHeader title="PUBLICATIONS" showArrows={true} />
      
      <ScrollView 
        horizontal 
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.pubScroll}
      >
        <PublicationCard
          icon="book"
          title="System Proposal"
          subtitle="Complete documentation..."
        />
        <PublicationCard
          icon="document"
          title="Research Paper"
          subtitle="Methodology & findings..."
        />
        <PublicationCard
          icon="analytics"
          title="Data Analysis"
          subtitle="Statistical reports..."
        />
        <PublicationCard
          icon="code-slash"
          title="Technical Docs"
          subtitle="Implementation guide..."
        />
        <PublicationCard
          icon="leaf"
          title="Talisay Guide"
          subtitle="About the fruit..."
        />
      </ScrollView>

      {/* ═══════════════════════════════════════════════════════════════════
          Info Cards Section
          ═══════════════════════════════════════════════════════════════════ */}
      <View style={styles.infoSection}>
        <View style={styles.infoCard}>
          <View style={styles.infoIconCircle}>
            <Ionicons name="color-palette" size={28} color="#fff" />
          </View>
          <Text style={styles.infoTitle}>Color Analysis</Text>
          <Text style={styles.infoDesc}>
            The system analyzes fruit color (green, yellow, brown) to estimate ripeness stage.
          </Text>
        </View>

        <View style={styles.infoCard}>
          <View style={[styles.infoIconCircle, { backgroundColor: '#f5a623' }]}>
            <Ionicons name="calculator" size={28} color="#fff" />
          </View>
          <Text style={styles.infoTitle}>Ratio Prediction</Text>
          <Text style={styles.infoDesc}>
            Predicts seed-to-oil ratio based on analyzed characteristics and ML model.
          </Text>
        </View>

        <View style={styles.infoCard}>
          <View style={[styles.infoIconCircle, { backgroundColor: '#1976d2' }]}>
            <Ionicons name="trending-up" size={28} color="#fff" />
          </View>
          <Text style={styles.infoTitle}>Expected Trend</Text>
          <Text style={styles.infoDesc}>
            Green fruits tend to have higher oil yield, followed by yellow, then brown.
          </Text>
        </View>
      </View>

      {/* Bottom Spacing */}
      <View style={{ height: 40 }} />
    </Screen>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   STYLES
   ══════════════════════════════════════════════════════════════════════════ */
const styles = StyleSheet.create({
  /* ──────────────────────────────────────────────────────────────────────────
     Hero Banner
     ────────────────────────────────────────────────────────────────────────── */
  heroBanner: {
    backgroundColor: theme.colors.greenDark,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 24,
    minHeight: 280,
    position: 'relative',
  },
  heroOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,102,51,0.85)',
  },
  heroContent: {
    padding: 32,
    alignItems: 'center',
    justifyContent: 'center',
  },
  heroTitle: {
    color: '#ffffff',
    fontSize: 42,
    fontWeight: '900',
    textAlign: 'center',
    letterSpacing: 2,
  },
  heroSubtitle: {
    color: '#ffffff',
    fontSize: 28,
    fontWeight: '300',
    textAlign: 'center',
    marginTop: 4,
  },
  heroTagline: {
    backgroundColor: 'rgba(255,255,255,0.15)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 25,
    marginTop: 20,
  },
  heroTaglineText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
  heroButtons: {
    flexDirection: 'row',
    gap: 16,
    marginTop: 28,
  },
  heroPrimaryBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: theme.colors.green,
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 8,
  },
  heroPrimaryBtnText: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '700',
  },
  heroSecondaryBtn: {
    backgroundColor: 'transparent',
    borderWidth: 2,
    borderColor: '#ffffff',
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 8,
  },
  heroSecondaryBtnText: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '700',
  },
  heroSocialRow: {
    position: 'absolute',
    right: 16,
    top: 16,
    flexDirection: 'row',
    gap: 8,
  },
  heroSocialBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },

  /* ──────────────────────────────────────────────────────────────────────────
     Section Header
     ────────────────────────────────────────────────────────────────────────── */
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 16,
    marginTop: 8,
  },
  sectionTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  sectionTitleLine: {
    flex: 1,
    height: 2,
    backgroundColor: theme.colors.green,
  },
  sectionTitle: {
    color: theme.colors.greenDark,
    fontSize: 18,
    fontWeight: '900',
    paddingHorizontal: 16,
    letterSpacing: 1,
  },
  sectionArrows: {
    flexDirection: 'row',
    gap: 4,
    marginLeft: 16,
  },
  arrowBtn: {
    width: 32,
    height: 32,
    borderRadius: 4,
    backgroundColor: '#e0e0e0',
    alignItems: 'center',
    justifyContent: 'center',
  },

  /* ──────────────────────────────────────────────────────────────────────────
     News Cards
     ────────────────────────────────────────────────────────────────────────── */
  newsScroll: {
    paddingBottom: 8,
    gap: 16,
  },
  newsCard: {
    width: 280,
    backgroundColor: '#ffffff',
    borderRadius: 8,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  newsImagePlaceholder: {
    height: 120,
    backgroundColor: '#f5f5f5',
    alignItems: 'center',
    justifyContent: 'center',
  },
  newsContent: {
    padding: 14,
  },
  newsTitle: {
    color: theme.colors.text,
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 8,
    lineHeight: 20,
  },
  newsSummary: {
    color: theme.colors.muted,
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 10,
  },
  newsReadMore: {
    color: theme.colors.green,
    fontSize: 13,
    fontWeight: '700',
  },

  /* ──────────────────────────────────────────────────────────────────────────
     Events Cards
     ────────────────────────────────────────────────────────────────────────── */
  eventsScroll: {
    paddingBottom: 8,
    gap: 16,
  },
  eventCard: {
    width: 220,
    backgroundColor: '#ffffff',
    borderRadius: 8,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  eventImageArea: {
    height: 80,
    backgroundColor: '#e8f5e9',
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  eventTitle: {
    color: theme.colors.text,
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 6,
  },
  eventDesc: {
    color: theme.colors.muted,
    fontSize: 12,
    lineHeight: 17,
    marginBottom: 10,
  },
  eventReadMore: {
    color: theme.colors.green,
    fontSize: 12,
    fontWeight: '700',
  },

  /* ──────────────────────────────────────────────────────────────────────────
     Publications Cards
     ────────────────────────────────────────────────────────────────────────── */
  pubScroll: {
    paddingBottom: 8,
    gap: 16,
    marginBottom: 24,
  },
  pubCard: {
    width: 160,
    backgroundColor: '#ffffff',
    borderRadius: 8,
    padding: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  pubIcon: {
    width: 70,
    height: 70,
    borderRadius: 8,
    backgroundColor: '#e8f5e9',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  pubTitle: {
    color: theme.colors.text,
    fontSize: 14,
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 4,
  },
  pubSubtitle: {
    color: theme.colors.muted,
    fontSize: 11,
    textAlign: 'center',
  },

  /* ──────────────────────────────────────────────────────────────────────────
     Quick Actions (Mobile)
     ────────────────────────────────────────────────────────────────────────── */
  quickActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 24,
  },
  quickActionBtn: {
    flex: 1,
    minWidth: 140,
    padding: 16,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  quickActionText: {
    color: theme.colors.text,
    fontSize: 13,
    fontWeight: '700',
  },

  /* ──────────────────────────────────────────────────────────────────────────
     Info Section
     ────────────────────────────────────────────────────────────────────────── */
  infoSection: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
  },
  infoCard: {
    flex: 1,
    minWidth: 260,
    backgroundColor: '#ffffff',
    borderRadius: 10,
    padding: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  infoIconCircle: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: theme.colors.green,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 14,
  },
  infoTitle: {
    color: theme.colors.text,
    fontSize: 16,
    fontWeight: '800',
    marginBottom: 8,
    textAlign: 'center',
  },
  infoDesc: {
    color: theme.colors.muted,
    fontSize: 13,
    lineHeight: 19,
    textAlign: 'center',
  },
});

export default memo(HomeScreen);