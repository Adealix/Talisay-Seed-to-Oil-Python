import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Screen from '../components/Screen';
import { useAuth } from '../hooks';
import { theme } from '../theme/theme';

export default function AccountScreen({ navigation }) {
  const { user, logout, isAdmin } = useAuth();

  if (!user) {
    return (
      <Screen scroll={false}>
        {/* Compact Header */}
        <View style={styles.header}>
          <View style={styles.headerIcon}>
            <Ionicons name="person" size={24} color="#fff" />
          </View>
          <View style={styles.headerText}>
            <Text style={styles.headerTitle}>Account</Text>
            <Text style={styles.headerSub}>Login to sync predictions</Text>
          </View>
        </View>

        {/* Login Options */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Get Started</Text>
          <Text style={styles.cardDesc}>Create an account to sync predictions to cloud.</Text>

          <Pressable style={styles.optionRow} onPress={() => navigation.navigate('Login')}>
            <View style={[styles.optionIcon, { backgroundColor: '#e8f5e9' }]}>
              <Ionicons name="log-in" size={18} color={theme.colors.green} />
            </View>
            <View style={styles.optionContent}>
              <Text style={styles.optionTitle}>Login</Text>
              <Text style={styles.optionDesc}>Already have an account</Text>
            </View>
            <Ionicons name="chevron-forward" size={16} color="#999" />
          </Pressable>

          <Pressable style={styles.optionRow} onPress={() => navigation.navigate('Register')}>
            <View style={[styles.optionIcon, { backgroundColor: '#e3f2fd' }]}>
              <Ionicons name="person-add" size={18} color="#1976d2" />
            </View>
            <View style={styles.optionContent}>
              <Text style={styles.optionTitle}>Create Account</Text>
              <Text style={styles.optionDesc}>New user? Register free</Text>
            </View>
            <Ionicons name="chevron-forward" size={16} color="#999" />
          </Pressable>
        </View>

        {/* Benefits */}
        <View style={[styles.card, styles.benefitsCard]}>
          <Text style={styles.cardTitle}>Benefits</Text>
          <View style={styles.benefitsRow}>
            <View style={styles.benefitItem}>
              <Ionicons name="cloud-upload" size={16} color={theme.colors.green} />
              <Text style={styles.benefitText}>Cloud Sync</Text>
            </View>
            <View style={styles.benefitItem}>
              <Ionicons name="phone-portrait-outline" size={16} color={theme.colors.green} />
              <Text style={styles.benefitText}>Multi-device</Text>
            </View>
            <View style={styles.benefitItem}>
              <Ionicons name="shield-checkmark" size={16} color={theme.colors.green} />
              <Text style={styles.benefitText}>Secure</Text>
            </View>
          </View>
        </View>
      </Screen>
    );
  }

  return (
    <Screen scroll={false}>
      {/* Profile Header */}
      <View style={styles.profileHeader}>
        <View style={styles.profileAvatar}>
          <Ionicons name="person" size={28} color="#fff" />
        </View>
        <View style={styles.profileInfo}>
          <Text style={styles.profileEmail} numberOfLines={1}>{user?.email || 'Unknown'}</Text>
          <View style={[styles.roleBadge, isAdmin && styles.roleBadgeAdmin]}>
            <Ionicons name={isAdmin ? 'shield' : 'person'} size={12} color={isAdmin ? '#fff' : '#333'} />
            <Text style={[styles.roleBadgeText, isAdmin && styles.roleBadgeTextAdmin]}>
              {user?.role?.toUpperCase() || 'USER'}
            </Text>
          </View>
        </View>
      </View>

      {/* Account Details */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Account Details</Text>
        
        <View style={styles.detailRow}>
          <View style={styles.detailIcon}>
            <Ionicons name="mail" size={16} color={theme.colors.green} />
          </View>
          <View style={styles.detailContent}>
            <Text style={styles.detailLabel}>Email</Text>
            <Text style={styles.detailValue} numberOfLines={1}>{user?.email || '-'}</Text>
          </View>
        </View>

        <View style={styles.detailRow}>
          <View style={styles.detailIcon}>
            <Ionicons name="shield" size={16} color={theme.colors.green} />
          </View>
          <View style={styles.detailContent}>
            <Text style={styles.detailLabel}>Role</Text>
            <Text style={styles.detailValue}>{user?.role || 'user'}</Text>
          </View>
        </View>

        <Pressable style={styles.logoutBtn} onPress={logout}>
          <Ionicons name="log-out" size={16} color={theme.colors.danger} />
          <Text style={styles.logoutText}>Logout</Text>
        </Pressable>
      </View>

      {/* Admin Section */}
      {isAdmin && (
        <View style={[styles.card, styles.adminCard]}>
          <View style={styles.adminHeader}>
            <View style={styles.adminIcon}>
              <Ionicons name="shield-checkmark" size={18} color="#fff" />
            </View>
            <View>
              <Text style={styles.adminTitle}>Admin Access</Text>
              <Text style={styles.adminSubtitle}>Manage users & predictions</Text>
            </View>
          </View>

          <View style={styles.adminBtnRow}>
            <Pressable style={styles.adminBtn} onPress={() => navigation.navigate('Admin')}>
              <Ionicons name="people" size={16} color="#fff" />
              <Text style={styles.adminBtnText}>Users</Text>
            </Pressable>
            <Pressable style={[styles.adminBtn, styles.adminBtnAlt]} onPress={() => navigation.navigate('AdminDashboard')}>
              <Ionicons name="analytics" size={16} color="#fff" />
              <Text style={styles.adminBtnText}>Analytics</Text>
            </Pressable>
          </View>
        </View>
      )}
    </Screen>
  );
}

const styles = StyleSheet.create({
  /* Header (Logged Out) */
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: theme.colors.greenDark,
    padding: 12,
    borderRadius: 8,
    marginBottom: 10,
  },
  headerIcon: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerText: { flex: 1 },
  headerTitle: { color: '#fff', fontSize: 16, fontWeight: '700' },
  headerSub: { color: 'rgba(255,255,255,0.8)', fontSize: 11, marginTop: 2 },

  /* Cards */
  card: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    marginBottom: 10,
  },
  cardTitle: {
    fontSize: 12,
    fontWeight: '800',
    color: theme.colors.greenDark,
    marginBottom: 8,
  },
  cardDesc: {
    fontSize: 11,
    color: '#666',
    marginBottom: 10,
  },

  /* Login Options */
  optionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    backgroundColor: '#f9f9f9',
    borderRadius: 8,
    marginBottom: 8,
    gap: 10,
  },
  optionIcon: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  optionContent: { flex: 1 },
  optionTitle: { fontSize: 12, fontWeight: '700', color: '#333' },
  optionDesc: { fontSize: 10, color: '#888', marginTop: 2 },

  /* Benefits */
  benefitsCard: { backgroundColor: '#e8f5e9' },
  benefitsRow: { flexDirection: 'row', gap: 12 },
  benefitItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  benefitText: { fontSize: 10, color: '#333' },

  /* Profile Header (Logged In) */
  profileHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: theme.colors.greenDark,
    padding: 12,
    borderRadius: 8,
    marginBottom: 10,
  },
  profileAvatar: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  profileInfo: { flex: 1 },
  profileEmail: { color: '#fff', fontSize: 14, fontWeight: '700' },
  roleBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(255,255,255,0.9)',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 10,
    alignSelf: 'flex-start',
    marginTop: 4,
  },
  roleBadgeAdmin: { backgroundColor: '#f5a623' },
  roleBadgeText: { fontSize: 10, fontWeight: '700', color: '#333' },
  roleBadgeTextAdmin: { color: '#fff' },

  /* Detail Rows */
  detailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  detailIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#e8f5e9',
    alignItems: 'center',
    justifyContent: 'center',
  },
  detailContent: { flex: 1 },
  detailLabel: { fontSize: 10, color: '#888' },
  detailValue: { fontSize: 12, fontWeight: '700', color: '#333' },

  /* Logout */
  logoutBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#ffebee',
    paddingVertical: 10,
    borderRadius: 8,
    marginTop: 10,
  },
  logoutText: { fontSize: 12, fontWeight: '700', color: theme.colors.danger },

  /* Admin */
  adminCard: { backgroundColor: '#fff3e0' },
  adminHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 10,
  },
  adminIcon: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#f5a623',
    alignItems: 'center',
    justifyContent: 'center',
  },
  adminTitle: { fontSize: 12, fontWeight: '800', color: '#e65100' },
  adminSubtitle: { fontSize: 10, color: '#8d6e63' },
  adminBtnRow: {
    flexDirection: 'row',
    gap: 8,
  },
  adminBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    backgroundColor: '#f5a623',
    paddingVertical: 10,
    borderRadius: 8,
  },
  adminBtnAlt: {
    backgroundColor: '#2196f3',
  },
  adminBtnText: { fontSize: 11, fontWeight: '700', color: '#fff' },
});
