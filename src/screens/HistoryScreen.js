import React, { useCallback, useEffect, useState } from 'react';
import { Alert, FlatList, Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Screen from '../components/Screen';
import { Body, Button, Card, Label, Title, Kicker } from '../components/Ui';
import { clearHistory, loadHistory } from '../storage';
import { theme } from '../theme/theme';

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function getCategoryColor(category) {
  switch (category) {
    case 'GREEN': return '#4caf50';
    case 'YELLOW': return '#ffc107';
    case 'BROWN': return '#8b4513';
    default: return theme.colors.muted;
  }
}

function getCategoryIcon(category) {
  switch (category) {
    case 'GREEN': return 'leaf';
    case 'YELLOW': return 'sunny';
    case 'BROWN': return 'ellipse';
    default: return 'help-circle';
  }
}

export default function HistoryScreen() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    const next = await loadHistory();
    setItems(next);
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function onClear() {
    if (Platform.OS === 'web') {
      const ok = window.confirm('Clear all history?');
      if (!ok) return;
    } else {
      const ok = await new Promise((resolve) => {
        Alert.alert('Clear History', 'Clear all saved predictions?', [
          { text: 'Cancel', style: 'cancel', onPress: () => resolve(false) },
          { text: 'Clear', style: 'destructive', onPress: () => resolve(true) },
        ]);
      });
      if (!ok) return;
    }

    await clearHistory();
    await refresh();
  }

  return (
    <Screen scroll={false} contentContainerStyle={{ paddingBottom: 0 }}>
      {/* Header */}
      <View style={styles.pageHeader}>
        <View style={styles.headerIcon}>
          <Ionicons name="time" size={32} color="#fff" />
        </View>
        <View style={styles.headerText}>
          <Title style={styles.headerTitle}>History</Title>
          <Body style={styles.headerSubtitle}>Saved predictions on your device</Body>
        </View>
      </View>

      {/* Actions */}
      <View style={styles.actionsRow}>
        <Pressable style={styles.actionBtn} onPress={refresh}>
          <Ionicons name="refresh" size={20} color={theme.colors.green} />
          <Text style={styles.actionBtnText}>Refresh</Text>
        </Pressable>
        <Pressable style={[styles.actionBtn, styles.actionBtnDanger]} onPress={onClear}>
          <Ionicons name="trash" size={20} color={theme.colors.danger} />
          <Text style={[styles.actionBtnText, styles.actionBtnTextDanger]}>Clear All</Text>
        </Pressable>
      </View>

      {/* Stats Summary */}
      {!loading && items.length > 0 && (
        <View style={styles.statsRow}>
          <View style={styles.statBox}>
            <Text style={styles.statNumber}>{items.length}</Text>
            <Text style={styles.statLabel}>Total Records</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statNumber}>
              {items.filter(i => i.category === 'GREEN').length}
            </Text>
            <Text style={styles.statLabel}>Green</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statNumber}>
              {items.filter(i => i.category === 'YELLOW').length}
            </Text>
            <Text style={styles.statLabel}>Yellow</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statNumber}>
              {items.filter(i => i.category === 'BROWN').length}
            </Text>
            <Text style={styles.statLabel}>Brown</Text>
          </View>
        </View>
      )}

      {/* List */}
      {loading ? (
        <Card style={styles.emptyCard}>
          <Ionicons name="hourglass" size={48} color="#ccc" />
          <Body style={styles.emptyText}>Loading...</Body>
        </Card>
      ) : items.length === 0 ? (
        <Card style={styles.emptyCard}>
          <Ionicons name="folder-open-outline" size={48} color="#ccc" />
          <Body style={styles.emptyText}>No saved results yet</Body>
          <Body style={styles.emptyHint}>Go to Scan and save a prediction</Body>
        </Card>
      ) : (
        <FlatList
          data={items}
          keyExtractor={(item) => item.id}
          contentContainerStyle={{ paddingBottom: 24 }}
          renderItem={({ item }) => (
            <Card style={styles.historyCard}>
              <View style={styles.cardHeader}>
                <View style={[styles.categoryBadge, { backgroundColor: getCategoryColor(item.category) }]}>
                  <Ionicons name={getCategoryIcon(item.category)} size={16} color="#fff" />
                  <Text style={styles.categoryBadgeText}>{item.category}</Text>
                </View>
                <Text style={styles.dateText}>{formatDate(item.createdAt)}</Text>
              </View>

              <View style={styles.ratioRow}>
                <View style={styles.ratioBox}>
                  <Text style={styles.ratioNumber}>{Math.round(item.ratio * 100)}%</Text>
                  <Text style={styles.ratioLabel}>Predicted Ratio</Text>
                </View>
                
                <View style={styles.inputsBox}>
                  <View style={styles.inputItem}>
                    <Ionicons name="resize" size={14} color={theme.colors.muted} />
                    <Text style={styles.inputText}>{item.inputs?.lengthMm ?? '-'} mm</Text>
                  </View>
                  <View style={styles.inputItem}>
                    <Ionicons name="swap-horizontal" size={14} color={theme.colors.muted} />
                    <Text style={styles.inputText}>{item.inputs?.widthMm ?? '-'} mm</Text>
                  </View>
                  <View style={styles.inputItem}>
                    <Ionicons name="scale" size={14} color={theme.colors.muted} />
                    <Text style={styles.inputText}>{item.inputs?.weightG ?? '-'} g</Text>
                  </View>
                </View>
              </View>
            </Card>
          )}
        />
      )}
    </Screen>
  );
}

const styles = StyleSheet.create({
  /* Page Header */
  pageHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    marginBottom: 16,
    padding: 20,
    backgroundColor: theme.colors.greenDark,
    borderRadius: 12,
  },
  headerIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
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

  /* Actions */
  actionsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  actionBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#e8f5e9',
    paddingVertical: 12,
    borderRadius: 8,
  },
  actionBtnDanger: {
    backgroundColor: '#ffebee',
  },
  actionBtnText: {
    color: theme.colors.green,
    fontWeight: '700',
  },
  actionBtnTextDanger: {
    color: theme.colors.danger,
  },

  /* Stats */
  statsRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 16,
  },
  statBox: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: theme.colors.border,
  },
  statNumber: {
    fontSize: 24,
    fontWeight: '900',
    color: theme.colors.greenDark,
  },
  statLabel: {
    fontSize: 11,
    color: theme.colors.muted,
    marginTop: 2,
  },

  /* Empty State */
  emptyCard: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  emptyText: {
    color: theme.colors.muted,
    marginTop: 12,
    fontSize: 16,
  },
  emptyHint: {
    color: theme.colors.light,
    marginTop: 4,
    fontSize: 13,
  },

  /* History Card */
  historyCard: {
    marginBottom: 12,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  categoryBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  categoryBadgeText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 12,
  },
  dateText: {
    color: theme.colors.muted,
    fontSize: 12,
  },
  ratioRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  ratioBox: {
    alignItems: 'center',
    paddingRight: 16,
    borderRightWidth: 1,
    borderRightColor: theme.colors.border,
  },
  ratioNumber: {
    fontSize: 32,
    fontWeight: '900',
    color: theme.colors.greenDark,
  },
  ratioLabel: {
    fontSize: 11,
    color: theme.colors.muted,
    marginTop: 2,
  },
  inputsBox: {
    flex: 1,
    gap: 6,
  },
  inputItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  inputText: {
    color: theme.colors.text,
    fontSize: 13,
  },
});
