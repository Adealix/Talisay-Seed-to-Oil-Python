import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Screen from '../components/Screen';
import { Body, Button, Card, Label, Title, Kicker, Divider } from '../components/Ui';
import { useAuth } from '../hooks';
import { theme } from '../theme/theme';

export default function AccountScreen({ navigation }) {
  const { user, logout, isAdmin } = useAuth();

  if (!user) {
    return (
      <Screen>
        {/* Header */}
        <View style={styles.pageHeader}>
          <View style={styles.headerIcon}>
            <Ionicons name="person" size={40} color="#fff" />
          </View>
          <View style={styles.headerText}>
            <Title style={styles.headerTitle}>Account</Title>
            <Body style={styles.headerSubtitle}>Login to sync predictions</Body>
          </View>
        </View>

        {/* Login Options */}
        <Card>
          <Kicker>Get Started</Kicker>
          <Body style={{ marginBottom: 16 }}>
            Create an account or login to sync your predictions to the cloud and access them from any device.
          </Body>

          <Pressable style={styles.loginOption} onPress={() => navigation.navigate('Login')}>
            <View style={styles.loginOptionIcon}>
              <Ionicons name="log-in" size={24} color={theme.colors.green} />
            </View>
            <View style={styles.loginOptionContent}>
              <Text style={styles.loginOptionTitle}>Login</Text>
              <Text style={styles.loginOptionDesc}>Already have an account? Sign in here</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={theme.colors.muted} />
          </Pressable>

          <Pressable style={styles.loginOption} onPress={() => navigation.navigate('Register')}>
            <View style={[styles.loginOptionIcon, { backgroundColor: '#e3f2fd' }]}>
              <Ionicons name="person-add" size={24} color="#1976d2" />
            </View>
            <View style={styles.loginOptionContent}>
              <Text style={styles.loginOptionTitle}>Create Account</Text>
              <Text style={styles.loginOptionDesc}>New user? Register for free</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={theme.colors.muted} />
          </Pressable>
        </Card>

        {/* Benefits */}
        <Card style={styles.benefitsCard}>
          <Kicker>Benefits of an Account</Kicker>
          <View style={styles.benefitsList}>
            <View style={styles.benefitItem}>
              <Ionicons name="cloud-upload" size={20} color={theme.colors.green} />
              <Body>Sync predictions to the cloud</Body>
            </View>
            <View style={styles.benefitItem}>
              <Ionicons name="devices" size={20} color={theme.colors.green} />
              <Body>Access from any device</Body>
            </View>
            <View style={styles.benefitItem}>
              <Ionicons name="shield-checkmark" size={20} color={theme.colors.green} />
              <Body>Secure data storage</Body>
            </View>
          </View>
        </Card>
      </Screen>
    );
  }

  return (
    <Screen>
      {/* Logged In Header */}
      <View style={styles.profileHeader}>
        <View style={styles.profileAvatar}>
          <Ionicons name="person" size={48} color="#fff" />
        </View>
        <View style={styles.profileInfo}>
          <Text style={styles.profileEmail}>{user?.email || 'Unknown'}</Text>
          <View style={[styles.roleBadge, isAdmin && styles.roleBadgeAdmin]}>
            <Ionicons 
              name={isAdmin ? 'shield' : 'person'} 
              size={14} 
              color={isAdmin ? '#fff' : theme.colors.text} 
            />
            <Text style={[styles.roleBadgeText, isAdmin && styles.roleBadgeTextAdmin]}>
              {user?.role?.toUpperCase() || 'USER'}
            </Text>
          </View>
        </View>
      </View>

      {/* Account Info Card */}
      <Card>
        <Kicker>Account Details</Kicker>
        
        <View style={styles.infoRow}>
          <View style={styles.infoIcon}>
            <Ionicons name="mail" size={20} color={theme.colors.green} />
          </View>
          <View style={styles.infoContent}>
            <Text style={styles.infoLabel}>Email</Text>
            <Text style={styles.infoValue}>{user?.email || '-'}</Text>
          </View>
        </View>

        <View style={styles.infoRow}>
          <View style={styles.infoIcon}>
            <Ionicons name="shield" size={20} color={theme.colors.green} />
          </View>
          <View style={styles.infoContent}>
            <Text style={styles.infoLabel}>Role</Text>
            <Text style={styles.infoValue}>{user?.role || 'user'}</Text>
          </View>
        </View>

        <Divider />

        <Pressable style={styles.logoutBtn} onPress={logout}>
          <Ionicons name="log-out" size={20} color={theme.colors.danger} />
          <Text style={styles.logoutBtnText}>Logout</Text>
        </Pressable>
      </Card>

      {/* Admin Section */}
      {isAdmin && (
        <Card style={styles.adminCard}>
          <View style={styles.adminHeader}>
            <View style={styles.adminIcon}>
              <Ionicons name="shield-checkmark" size={24} color="#fff" />
            </View>
            <View>
              <Text style={styles.adminTitle}>Admin Access</Text>
              <Text style={styles.adminSubtitle}>Manage users and view all predictions</Text>
            </View>
          </View>

          <Pressable 
            style={styles.adminBtn} 
            onPress={() => navigation.navigate('Admin')}
          >
            <Ionicons name="settings" size={20} color="#fff" />
            <Text style={styles.adminBtnText}>Open Admin Dashboard</Text>
            <Ionicons name="arrow-forward" size={20} color="#fff" />
          </Pressable>
        </Card>
      )}
    </Screen>
  );
}

const styles = StyleSheet.create({
  /* Page Header (Logged Out) */
  pageHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    marginBottom: 20,
    padding: 24,
    backgroundColor: theme.colors.greenDark,
    borderRadius: 12,
  },
  headerIcon: {
    width: 72,
    height: 72,
    borderRadius: 36,
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

  /* Login Options */
  loginOption: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    backgroundColor: '#f9f9f9',
    borderRadius: 10,
    marginBottom: 10,
    gap: 14,
  },
  loginOptionIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#e8f5e9',
    alignItems: 'center',
    justifyContent: 'center',
  },
  loginOptionContent: {
    flex: 1,
  },
  loginOptionTitle: {
    color: theme.colors.text,
    fontWeight: '700',
    fontSize: 15,
  },
  loginOptionDesc: {
    color: theme.colors.muted,
    fontSize: 12,
    marginTop: 2,
  },

  /* Benefits */
  benefitsCard: {
    backgroundColor: '#e8f5e9',
    borderColor: theme.colors.green,
  },
  benefitsList: {
    gap: 12,
    marginTop: 8,
  },
  benefitItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },

  /* Profile Header (Logged In) */
  profileHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    marginBottom: 20,
    padding: 24,
    backgroundColor: theme.colors.greenDark,
    borderRadius: 12,
  },
  profileAvatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  profileInfo: {
    flex: 1,
  },
  profileEmail: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 8,
  },
  roleBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: 'rgba(255,255,255,0.9)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    alignSelf: 'flex-start',
  },
  roleBadgeAdmin: {
    backgroundColor: '#f5a623',
  },
  roleBadgeText: {
    color: theme.colors.text,
    fontWeight: '700',
    fontSize: 12,
  },
  roleBadgeTextAdmin: {
    color: '#fff',
  },

  /* Info Rows */
  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    paddingVertical: 12,
  },
  infoIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#e8f5e9',
    alignItems: 'center',
    justifyContent: 'center',
  },
  infoContent: {
    flex: 1,
  },
  infoLabel: {
    color: theme.colors.muted,
    fontSize: 12,
    fontWeight: '600',
  },
  infoValue: {
    color: theme.colors.text,
    fontSize: 15,
    fontWeight: '700',
    marginTop: 2,
  },

  /* Logout */
  logoutBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    backgroundColor: '#ffebee',
    paddingVertical: 14,
    borderRadius: 10,
  },
  logoutBtnText: {
    color: theme.colors.danger,
    fontWeight: '700',
    fontSize: 15,
  },

  /* Admin Card */
  adminCard: {
    backgroundColor: '#fff3e0',
    borderColor: '#f5a623',
  },
  adminHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    marginBottom: 16,
  },
  adminIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#f5a623',
    alignItems: 'center',
    justifyContent: 'center',
  },
  adminTitle: {
    color: '#e65100',
    fontWeight: '800',
    fontSize: 16,
  },
  adminSubtitle: {
    color: '#8d6e63',
    fontSize: 12,
    marginTop: 2,
  },
  adminBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    backgroundColor: '#f5a623',
    paddingVertical: 14,
    borderRadius: 10,
  },
  adminBtnText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 15,
    flex: 1,
    textAlign: 'center',
  },
});
