import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { theme } from '../theme/theme';

/* ══════════════════════════════════════════════════════════════════════════
   UI COMPONENTS - UCAP-Styled Design System
   ══════════════════════════════════════════════════════════════════════════ */

export function Card({ children, style }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

export function Title({ children, style }) {
  return <Text style={[styles.title, style]}>{children}</Text>;
}

export function Subtitle({ children, style }) {
  return <Text style={[styles.subtitle, style]}>{children}</Text>;
}

export function Label({ children, style }) {
  return <Text style={[styles.label, style]}>{children}</Text>;
}

export function Body({ children, style }) {
  return <Text style={[styles.body, style]}>{children}</Text>;
}

export function Kicker({ children, style }) {
  return <Text style={[styles.kicker, style]}>{children}</Text>;
}

export function SectionTitle({ children, style }) {
  return (
    <View style={styles.sectionTitleWrapper}>
      <View style={styles.sectionTitleLine} />
      <Text style={[styles.sectionTitleText, style]}>{children}</Text>
      <View style={styles.sectionTitleLine} />
    </View>
  );
}

export function Button({ title, onPress, disabled = false, variant = 'primary', style, icon }) {
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={({ pressed, hovered }) => [
        styles.button,
        variant === 'secondary' ? styles.buttonSecondary : 
        variant === 'outline' ? styles.buttonOutline :
        variant === 'danger' ? styles.buttonDanger :
        styles.buttonPrimary,
        disabled && styles.buttonDisabled,
        pressed && !disabled && styles.buttonPressed,
        hovered && !disabled && styles.buttonHover,
        style,
      ]}
    >
      <Text style={[
        styles.buttonText, 
        variant === 'secondary' && styles.buttonTextSecondary,
        variant === 'outline' && styles.buttonTextOutline,
      ]}>
        {title}
      </Text>
    </Pressable>
  );
}

export function Badge({ children, variant = 'default', style }) {
  return (
    <View style={[
      styles.badge,
      variant === 'success' && styles.badgeSuccess,
      variant === 'warning' && styles.badgeWarning,
      variant === 'danger' && styles.badgeDanger,
      style,
    ]}>
      <Text style={[
        styles.badgeText,
        variant === 'success' && styles.badgeTextSuccess,
        variant === 'warning' && styles.badgeTextWarning,
        variant === 'danger' && styles.badgeTextDanger,
      ]}>{children}</Text>
    </View>
  );
}

export function Divider({ style }) {
  return <View style={[styles.divider, style]} />;
}

/* ══════════════════════════════════════════════════════════════════════════
   STYLES
   ══════════════════════════════════════════════════════════════════════════ */
const styles = StyleSheet.create({
  /* Card */
  card: {
    backgroundColor: theme.colors.cardBg,
    borderColor: theme.colors.border,
    borderWidth: 1,
    borderRadius: 10,
    padding: 20,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
    elevation: 2,
  },

  /* Typography */
  title: {
    color: theme.colors.text,
    fontSize: 24,
    fontWeight: '800',
    marginBottom: 8,
    letterSpacing: 0.3,
  },
  subtitle: {
    color: theme.colors.muted,
    fontSize: 15,
    marginBottom: 12,
    lineHeight: 22,
  },
  label: {
    color: theme.colors.muted,
    fontSize: 12,
    fontWeight: '800',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 6,
  },
  body: {
    color: theme.colors.text,
    fontSize: 14,
    lineHeight: 22,
  },
  kicker: {
    color: theme.colors.greenDark,
    fontSize: 12,
    fontWeight: '900',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 8,
  },

  /* Section Title */
  sectionTitleWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 16,
  },
  sectionTitleLine: {
    flex: 1,
    height: 2,
    backgroundColor: theme.colors.green,
  },
  sectionTitleText: {
    color: theme.colors.greenDark,
    fontSize: 18,
    fontWeight: '900',
    paddingHorizontal: 16,
    letterSpacing: 1,
  },

  /* Button */
  button: {
    borderRadius: 8,
    paddingVertical: 14,
    paddingHorizontal: 20,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    flexDirection: 'row',
    gap: 8,
  },
  buttonPrimary: {
    backgroundColor: theme.colors.green,
    borderColor: theme.colors.green,
  },
  buttonSecondary: {
    backgroundColor: '#ffffff',
    borderColor: theme.colors.border,
  },
  buttonOutline: {
    backgroundColor: 'transparent',
    borderColor: theme.colors.green,
  },
  buttonDanger: {
    backgroundColor: theme.colors.danger,
    borderColor: theme.colors.danger,
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonPressed: {
    transform: [{ scale: 0.98 }],
    opacity: 0.9,
  },
  buttonHover: {
    opacity: 0.9,
  },
  buttonText: {
    color: '#ffffff',
    fontWeight: '800',
    fontSize: 14,
    letterSpacing: 0.3,
  },
  buttonTextSecondary: {
    color: theme.colors.text,
  },
  buttonTextOutline: {
    color: theme.colors.green,
  },

  /* Badge */
  badge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: '#e0e0e0',
    alignSelf: 'flex-start',
  },
  badgeSuccess: {
    backgroundColor: '#e8f5e9',
  },
  badgeWarning: {
    backgroundColor: '#fff3e0',
  },
  badgeDanger: {
    backgroundColor: '#ffebee',
  },
  badgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#666',
  },
  badgeTextSuccess: {
    color: '#2e7d32',
  },
  badgeTextWarning: {
    color: '#f57c00',
  },
  badgeTextDanger: {
    color: '#c62828',
  },

  /* Divider */
  divider: {
    height: 1,
    backgroundColor: theme.colors.border,
    marginVertical: 16,
  },
});
