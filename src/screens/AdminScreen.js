import React, { useEffect, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import Screen from '../components/Screen';
import { Card, Title, Body, Kicker, Badge } from '../components/Ui';
import { useAuth } from '../hooks';
import { adminService } from '../services';
import { theme } from '../theme/theme';

export default function AdminScreen({ navigation }) {
  const { isAdmin } = useAuth();
  const [users, setUsers] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);

  async function load() {
    setLoading(true);
    try {
      const [usersList, predictionsList] = await Promise.all([
        adminService.listUsers(),
        adminService.listAllPredictions(),
      ]);
      setUsers(usersList);
      setPredictions(predictionsList);
    } catch (e) {
      console.warn('[AdminScreen.load]', e?.message);
      setUsers([]);
      setPredictions([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (isAdmin) load();
  }, [isAdmin]);

  // Not admin
  if (!isAdmin) {
    return (
      <Screen>
        <View style={styles.accessDenied}>
          <View style={styles.deniedIcon}>
            <Ionicons name="lock-closed" size={48} color={theme.colors.danger} />
          </View>
          <Title style={styles.deniedTitle}>Access Denied</Title>
          <Body style={styles.deniedText}>
            This page is restricted to admin accounts only.
          </Body>
          <Pressable 
            style={styles.goBackBtn}
            onPress={() => navigation.navigate('Account')}
          >
            <Ionicons name="arrow-back" size={18} color="#fff" />
            <Text style={styles.goBackBtnText}>Go to Account</Text>
          </Pressable>
        </View>
      </Screen>
    );
  }

  return (
    <Screen>
      <ScrollView contentContainerStyle={{ paddingBottom: 24 }}>
        {/* Header */}
        <View style={styles.pageHeader}>
          <View style={styles.headerIcon}>
            <Ionicons name="shield-checkmark" size={40} color="#fff" />
          </View>
          <View style={styles.headerContent}>
            <Title style={styles.headerTitle}>Admin Dashboard</Title>
            <Body style={styles.headerSubtitle}>Manage users and predictions</Body>
          </View>
        </View>

        {/* Stats Row */}
        <View style={styles.statsRow}>
          <View style={styles.statBox}>
            <Ionicons name="people" size={24} color={theme.colors.green} />
            <Text style={styles.statNumber}>{users.length}</Text>
            <Text style={styles.statLabel}>Users</Text>
          </View>
          <View style={styles.statBox}>
            <Ionicons name="analytics" size={24} color={theme.colors.orange} />
            <Text style={styles.statNumber}>{predictions.length}</Text>
            <Text style={styles.statLabel}>Predictions</Text>
          </View>
          <Pressable 
            style={[styles.statBox, styles.refreshBox]}
            onPress={load}
            disabled={loading}
          >
            <Ionicons 
              name={loading ? "hourglass" : "refresh"} 
              size={24} 
              color="#fff" 
            />
            <Text style={styles.refreshLabel}>
              {loading ? 'Loading...' : 'Refresh'}
            </Text>
          </Pressable>
        </View>

        {/* Users Section */}
        <Card>
          <View style={styles.sectionHeader}>
            <View style={styles.sectionIconWrap}>
              <Ionicons name="people" size={20} color="#fff" />
            </View>
            <Kicker style={styles.sectionTitle}>Registered Users</Kicker>
          </View>

          {users.length === 0 ? (
            <View style={styles.emptyState}>
              <Ionicons name="person-outline" size={32} color={theme.colors.muted} />
              <Body style={styles.emptyText}>No users found</Body>
            </View>
          ) : (
            users.map((u, index) => (
              <View 
                key={u.id} 
                style={[styles.userRow, index < users.length - 1 && styles.userRowBorder]}
              >
                <View style={styles.userAvatar}>
                  <Ionicons 
                    name={u.role === 'admin' ? 'shield' : 'person'} 
                    size={20} 
                    color={u.role === 'admin' ? theme.colors.orange : theme.colors.green} 
                  />
                </View>
                <View style={styles.userInfo}>
                  <Text style={styles.userEmail}>{u.email}</Text>
                  <Badge 
                    variant={u.role === 'admin' ? 'warning' : 'success'}
                  >
                    {u.role?.toUpperCase()}
                  </Badge>
                </View>
              </View>
            ))
          )}
        </Card>

        {/* Predictions Section */}
        <Card>
          <View style={styles.sectionHeader}>
            <View style={[styles.sectionIconWrap, { backgroundColor: theme.colors.orange }]}>
              <Ionicons name="analytics" size={20} color="#fff" />
            </View>
            <Kicker style={styles.sectionTitle}>Latest Predictions</Kicker>
          </View>

          {predictions.length === 0 ? (
            <View style={styles.emptyState}>
              <Ionicons name="flask-outline" size={32} color={theme.colors.muted} />
              <Body style={styles.emptyText}>No predictions found</Body>
            </View>
          ) : (
            predictions.slice(0, 20).map((p, index) => (
              <View 
                key={p._id} 
                style={[styles.predRow, index < Math.min(predictions.length, 20) - 1 && styles.predRowBorder]}
              >
                <View style={[
                  styles.predCategory,
                  p.category === 'green' && { backgroundColor: '#4caf50' },
                  p.category === 'yellow' && { backgroundColor: '#ffc107' },
                  p.category === 'brown' && { backgroundColor: '#795548' },
                ]}>
                  <Ionicons name="leaf" size={16} color="#fff" />
                </View>
                <View style={styles.predInfo}>
                  <Text style={styles.predCategoryLabel}>
                    {p.category?.charAt(0).toUpperCase() + p.category?.slice(1)} Talisay
                  </Text>
                  <Text style={styles.predRatio}>
                    {p.oilYieldPercent 
                      ? `Oil Yield: ${p.oilYieldPercent.toFixed(1)}%` 
                      : `Ratio: ${p.ratio || 'N/A'}`}
                  </Text>
                </View>
                <View style={styles.predUser}>
                  <Ionicons name="person-circle-outline" size={16} color={theme.colors.muted} />
                  <Text style={styles.predUserId} numberOfLines={1}>
                    {p.userEmail || String(p.userId).substring(0, 8) + '...'}
                  </Text>
                </View>
              </View>
            ))
          )}
        </Card>

        {/* Info Card */}
        <Card style={styles.infoCard}>
          <View style={styles.infoRow}>
            <Ionicons name="information-circle" size={20} color={theme.colors.orange} />
            <Body style={styles.infoText}>
              Showing up to 20 most recent predictions. This dashboard is for admin monitoring purposes.
            </Body>
          </View>
        </Card>
      </ScrollView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  /* Access Denied */
  accessDenied: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
    backgroundColor: '#fff',
    borderRadius: 16,
  },
  deniedIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#ffebee',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  deniedTitle: {
    color: theme.colors.danger,
    marginBottom: 8,
  },
  deniedText: {
    textAlign: 'center',
    marginBottom: 24,
  },
  goBackBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: theme.colors.green,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  goBackBtnText: {
    color: '#fff',
    fontWeight: '700',
  },

  /* Page Header */
  pageHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    marginBottom: 20,
    padding: 24,
    backgroundColor: theme.colors.orange,
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
  headerContent: {
    flex: 1,
  },
  headerTitle: {
    color: '#ffffff',
    marginBottom: 0,
  },
  headerSubtitle: {
    color: 'rgba(255,255,255,0.9)',
    marginTop: 4,
  },

  /* Stats Row */
  statsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  statBox: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  statNumber: {
    fontSize: 28,
    fontWeight: '800',
    color: theme.colors.text,
    marginTop: 4,
  },
  statLabel: {
    fontSize: 12,
    color: theme.colors.muted,
    fontWeight: '600',
  },
  refreshBox: {
    backgroundColor: theme.colors.green,
    justifyContent: 'center',
  },
  refreshLabel: {
    color: '#fff',
    fontWeight: '700',
    marginTop: 4,
    fontSize: 12,
  },

  /* Section Header */
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 16,
  },
  sectionIconWrap: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: theme.colors.green,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sectionTitle: {
    marginBottom: 0,
  },

  /* Empty State */
  emptyState: {
    alignItems: 'center',
    padding: 32,
  },
  emptyText: {
    marginTop: 8,
    color: theme.colors.muted,
  },

  /* User Row */
  userRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
  },
  userRowBorder: {
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.border,
  },
  userAvatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#f3f4f6',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  userInfo: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  userEmail: {
    fontSize: 14,
    fontWeight: '600',
    color: theme.colors.text,
    flex: 1,
  },

  /* Prediction Row */
  predRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
  },
  predRowBorder: {
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.border,
  },
  predCategory: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: theme.colors.green,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  predInfo: {
    flex: 1,
  },
  predCategoryLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: theme.colors.text,
  },
  predRatio: {
    fontSize: 12,
    color: theme.colors.muted,
    marginTop: 2,
  },
  predUser: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  predUserId: {
    fontSize: 11,
    color: theme.colors.muted,
    maxWidth: 80,
  },

  /* Info Card */
  infoCard: {
    backgroundColor: '#fff8e1',
    borderColor: theme.colors.orange,
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
  },
  infoText: {
    flex: 1,
    color: '#e65100',
  },
});
