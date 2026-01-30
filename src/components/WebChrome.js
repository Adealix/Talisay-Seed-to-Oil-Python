import React from 'react';
import { Image, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, useWindowDimensions, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute } from '@react-navigation/native';
import { useAuth } from '../hooks';
import { theme } from '../theme/theme';

/* ──────────────────────────────────────────────────────────────────────────
   Main WebChrome Component - UCAP Layout (Simplified)
   ────────────────────────────────────────────────────────────────────────── */
export default function WebChrome({ children }) {
  const navigation = useNavigation();
  const route = useRoute();
  const { width } = useWindowDimensions();
  const { user, isAuthenticated, isAdmin, logout } = useAuth();

  const isDesktop = width >= 1024;
  const isMobile = width < 768;

  // Get current route name for nav highlighting
  const currentRoute = route?.name || 'Home';

  // Tab screens that need nested navigation
  const tabScreens = ['Home', 'Scan', 'History', 'Proposal', 'Account'];

  // Simple navigation function - handles both tab and stack screens
  const goTo = (screen) => {
    if (tabScreens.includes(screen)) {
      // Navigate to tab screen via Tabs navigator
      navigation.navigate('Tabs', { screen });
    } else {
      // Navigate directly to stack screen
      navigation.navigate(screen);
    }
  };

  // Only intended for web
  if (Platform.OS !== 'web') return <>{children}</>;

  // Navigation items array - only essential screens
  const navItems = [
    { label: 'HOME', route: 'Home' },
    { label: 'ABOUT US', route: 'AboutUs' },
    { label: 'SCAN', route: 'Scan' },
    { label: 'ACCOUNT', route: 'Account' },
    // Commented out for now:
    // { label: 'PUBLICATIONS', route: 'Publications' },
    // { label: 'DIRECTORY', route: 'Directory' },
    // { label: 'NEWS', route: 'News' },
    // { label: 'EVENTS', route: 'Events' },
  ];

  return (
    <View style={styles.page}>
      {/* ════════════════════════════════════════════════════════════════════
          TOP HEADER BAR - Dark Green with Logo
          ════════════════════════════════════════════════════════════════════ */}
      <View style={styles.topHeader}>
        <View style={[styles.topHeaderInner, isMobile && styles.topHeaderInnerMobile]}>
          {/* Logo + Title Section - Clickable to go Home */}
          <Pressable style={styles.brandSection} onPress={() => goTo('Home')}>
            <View style={styles.logoWrapper}>
              <View style={styles.logoBg}>
                <Ionicons name="leaf" size={40} color="#ffffff" />
              </View>
            </View>
            <View style={styles.titleSection}>
              <Text style={[styles.mainTitle, isMobile && styles.mainTitleMobile]}>
                Talisay Oil Yield Predictor
              </Text>
              <Text style={styles.subtitle}>
                Machine Learning System Proposal
              </Text>
            </View>
          </Pressable>

          {/* User Status + Social Icons */}
          {!isMobile && (
            <View style={styles.headerRightSection}>
              {isAuthenticated ? (
                <Pressable style={styles.userIndicator} onPress={() => goTo('Account')}>
                  <View style={[styles.userAvatar, isAdmin && styles.userAvatarAdmin]}>
                    <Ionicons name={isAdmin ? 'shield' : 'person'} size={16} color="#fff" />
                  </View>
                  <Text style={styles.userEmail} numberOfLines={1}>
                    {user?.email?.split('@')[0]}
                  </Text>
                  <Pressable style={styles.logoutIconBtn} onPress={logout}>
                    <Ionicons name="log-out" size={16} color="rgba(255,255,255,0.7)" />
                  </Pressable>
                </Pressable>
              ) : (
                <Pressable style={styles.loginBtn} onPress={() => goTo('Login')}>
                  <Ionicons name="log-in" size={16} color="#fff" />
                  <Text style={styles.loginBtnText}>Login</Text>
                </Pressable>
              )}
              <View style={styles.socialSection}>
                <View style={styles.socialIcon}><Ionicons name="logo-facebook" size={20} color="#ffffff" /></View>
                <View style={styles.socialIcon}><Ionicons name="logo-twitter" size={20} color="#ffffff" /></View>
              </View>
            </View>
          )}
        </View>
      </View>

      {/* ════════════════════════════════════════════════════════════════════
          NAVIGATION BAR - Simple inline buttons (no ScrollView)
          ════════════════════════════════════════════════════════════════════ */}
      <View style={styles.navBar}>
        <View style={styles.navLinksContainer}>
          {navItems.map((item) => (
            <Pressable
              key={item.route}
              onPress={() => goTo(item.route)}
              style={[
                styles.navLink,
                currentRoute === item.route && styles.navLinkActive,
              ]}
            >
              <Text style={[
                styles.navLinkText,
                currentRoute === item.route && styles.navLinkTextActive,
              ]}>
                {item.label}
              </Text>
            </Pressable>
          ))}
        </View>
      </View>

      {/* ════════════════════════════════════════════════════════════════════
          SCROLLABLE CONTENT AREA - Contains Sidebar + Content + Footer
          ════════════════════════════════════════════════════════════════════ */}
      <ScrollView 
        style={styles.scrollContainer}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={true}
      >
        {/* Main Content Row: Sidebar + Content */}
        <View style={styles.contentWrapper}>
          <View style={[styles.contentInner, isMobile && styles.contentInnerMobile]}>
            {/* Left Sidebar - Only on desktop */}
            {isDesktop && (
              <View style={styles.sidebar}>
                <Pressable style={styles.sideButton} onPress={() => goTo('AboutUs')}>
                  <View style={styles.sideIconCircle}><Ionicons name="information-circle" size={24} color="#ffffff" /></View>
                  <Text style={styles.sideLabel}>ABOUT US</Text>
                </Pressable>
                <Pressable style={styles.sideButton} onPress={() => goTo('Scan')}>
                  <View style={styles.sideIconCircle}><Ionicons name="scan" size={24} color="#ffffff" /></View>
                  <Text style={styles.sideLabel}>SCAN FRUIT</Text>
                </Pressable>
                <Pressable style={styles.sideButton} onPress={() => goTo('Account')}>
                  <View style={styles.sideIconCircle}><Ionicons name="person" size={24} color="#ffffff" /></View>
                  <Text style={styles.sideLabel}>MY ACCOUNT</Text>
                </Pressable>
              </View>
            )}

            {/* Main Content Area */}
            <View style={[styles.mainContent, !isDesktop && styles.mainContentFull]}>
              {children}
            </View>
          </View>
        </View>

        {/* ════════════════════════════════════════════════════════════════════
            FOOTER
            ════════════════════════════════════════════════════════════════════ */}
        <View style={styles.footer}>
          <View style={[styles.footerInner, isMobile && styles.footerInnerMobile]}>
            <View style={styles.footerLogoSection}>
              <View style={styles.footerLogo}>
                <Ionicons name="leaf" size={32} color="#2a9d5c" />
              </View>
              <View style={styles.footerTitleSection}>
                <Text style={styles.footerTitle}>Talisay Oil Yield Predictor</Text>
                <Text style={styles.footerSubtitle}>Machine Learning System Proposal</Text>
              </View>
            </View>
            <View style={styles.footerContact}>
              <Text style={styles.footerHeading}>CONTACT</Text>
              <Text style={styles.footerText}>School System Proposal Project</Text>
              <Text style={styles.footerText}>Email: contact@talisayml.example.com</Text>
            </View>
          </View>
          <View style={styles.footerBottom}>
            <Text style={styles.footerCopyright}>
              Design inspired by UCAP website layout. This is a prototype for educational purposes.
            </Text>
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

/* ════════════════════════════════════════════════════════════════════════════
   STYLES - Matching UCAP Website Design
   ════════════════════════════════════════════════════════════════════════════ */
const styles = StyleSheet.create({
  /* ──────────────────────────────────────────────────────────────────────────
     Page Container
     ────────────────────────────────────────────────────────────────────────── */
  page: {
    flex: 1,
    backgroundColor: theme.colors.pageBg,
    height: '100%',
  },

  /* ──────────────────────────────────────────────────────────────────────────
     Scroll Container
     ────────────────────────────────────────────────────────────────────────── */
  scrollContainer: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },

  /* ──────────────────────────────────────────────────────────────────────────
     Top Header (Dark Green)
     ────────────────────────────────────────────────────────────────────────── */
  topHeader: {
    backgroundColor: theme.colors.greenDark,
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: 16,
  },
  topHeaderInner: {
    maxWidth: theme.maxWidth,
    width: '100%',
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  topHeaderInnerMobile: {
    justifyContent: 'center',
  },

  /* Brand Section */
  brandSection: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    cursor: 'pointer',
  },
  logoWrapper: {
    width: 70,
    height: 70,
  },
  logoBg: {
    width: 70,
    height: 70,
    borderRadius: 12,
    backgroundColor: theme.colors.green,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.2)',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },

  /* Title Section */
  titleSection: {
    marginLeft: 4,
  },
  mainTitle: {
    color: '#ffffff',
    fontSize: 26,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  mainTitleMobile: {
    fontSize: 18,
    textAlign: 'center',
  },
  subtitle: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 12,
    fontWeight: '500',
    marginTop: 2,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },

  /* Header Right Section */
  headerRightSection: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },

  /* User Indicator (Logged In) */
  userIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: 'rgba(255,255,255,0.15)',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 20,
  },
  userAvatar: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: theme.colors.green,
    alignItems: 'center',
    justifyContent: 'center',
  },
  userAvatarAdmin: {
    backgroundColor: theme.colors.orange,
  },
  userEmail: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '600',
    maxWidth: 120,
  },
  logoutIconBtn: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: 'rgba(255,255,255,0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 4,
  },

  /* Login Button (Logged Out) */
  loginBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: theme.colors.green,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 20,
  },
  loginBtnText: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '700',
  },

  /* Social Icons */
  socialSection: {
    flexDirection: 'row',
    gap: 8,
  },
  socialIcon: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(255,255,255,0.15)',
    alignItems: 'center',
    justifyContent: 'center',
  },

  /* ──────────────────────────────────────────────────────────────────────────
     Navigation Bar (Dark Gray)
     ────────────────────────────────────────────────────────────────────────── */
  navBar: {
    backgroundColor: theme.colors.navDark,
    paddingVertical: 4,
  },
  navLinksContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    flexWrap: 'wrap',
  },

  /* Nav Link */
  navLink: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    marginHorizontal: 4,
  },
  navLinkActive: {
    backgroundColor: theme.colors.green,
  },
  navLinkText: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  navLinkTextActive: {
    color: '#ffffff',
  },

  /* Search Box - commented out for now */
  /* Search Box - commented out for now
  searchBox: {
    flexDirection: 'row',
    backgroundColor: '#ffffff',
    borderRadius: 4,
    overflow: 'hidden',
    width: 240,
    marginLeft: 'auto',
    marginVertical: 8,
  },
  searchInput: {
    flex: 1,
    paddingHorizontal: 12,
    paddingVertical: 8,
    fontSize: 13,
    color: '#333',
  },
  searchButton: {
    width: 40,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f5f5f5',
    borderLeftWidth: 1,
    borderLeftColor: '#ddd',
  },
  */

  /* ──────────────────────────────────────────────────────────────────────────
     Content Area
     ────────────────────────────────────────────────────────────────────────── */
  contentWrapper: {
    flexGrow: 1,
    paddingHorizontal: 20,
    paddingTop: 24,
    paddingBottom: 40,
    minHeight: 400,
  },
  contentInner: {
    maxWidth: theme.maxWidth,
    width: '100%',
    alignSelf: 'center',
    flexDirection: 'row',
    gap: 20,
  },
  contentInnerMobile: {
    flexDirection: 'column',
  },

  /* Sidebar */
  sidebar: {
    width: 260,
    gap: 12,
    paddingTop: 16,
    overflow: 'hidden',
  },
  sideButton: {
    backgroundColor: theme.colors.green,
    borderRadius: 8,
    paddingVertical: 16,
    paddingHorizontal: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 4,
    elevation: 3,
  },
  sideButtonHover: {
    backgroundColor: '#25914f',
  },
  sideButtonImage: {
    width: '100%',
    height: 100,
    borderRadius: 6,
  },
  sideIconCircle: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0,0,0,0.15)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  sideLabel: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '800',
    letterSpacing: 0.5,
    flex: 1,
  },

  /* Main Content */
  mainContent: {
    flex: 1,
    minWidth: 0,
  },
  mainContentFull: {
    maxWidth: '100%',
  },

  /* ──────────────────────────────────────────────────────────────────────────
     Footer
     ────────────────────────────────────────────────────────────────────────── */
  footer: {
    backgroundColor: '#2d2d2d',
    paddingTop: 40,
  },
  footerInner: {
    maxWidth: theme.maxWidth,
    width: '100%',
    alignSelf: 'center',
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingBottom: 30,
    gap: 40,
  },
  footerInnerMobile: {
    flexDirection: 'column',
    gap: 30,
  },

  /* Footer Logo */
  footerLogoSection: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    flex: 1,
  },
  footerLogo: {
    width: 70,
    height: 70,
    borderRadius: 8,
    backgroundColor: '#ffffff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  footerTitleSection: {},
  footerTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '700',
  },
  footerSubtitle: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 12,
    marginTop: 4,
  },

  /* Footer Contact */
  footerContact: {
    flex: 1,
  },
  footerHeading: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '800',
    marginBottom: 12,
    letterSpacing: 0.5,
  },
  footerText: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 13,
    lineHeight: 22,
  },

  /* Footer Social */
  footerSocial: {},
  footerSocialIcons: {
    flexDirection: 'row',
    gap: 8,
  },
  footerSocialIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#3b5998',
    alignItems: 'center',
    justifyContent: 'center',
  },

  /* Footer Bottom */
  footerBottom: {
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.1)',
    paddingVertical: 16,
    paddingHorizontal: 20,
  },
  footerCopyright: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 12,
    textAlign: 'center',
  },
});
