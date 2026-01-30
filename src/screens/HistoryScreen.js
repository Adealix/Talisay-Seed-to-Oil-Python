import React, { useCallback, useEffect, useState } from 'react';
import { Alert, FlatList, Platform, Pressable, StyleSheet, Text, View, useWindowDimensions } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Screen from '../components/Screen';
import { clearHistory as clearLocalHistory, loadHistory as loadLocalHistory } from '../storage';
import { historyService } from '../services';
import { useAuth } from '../hooks';
import { theme } from '../theme/theme';

const MOBILE_BREAKPOINT = 600;

function formatDate(iso) {
  try {
    const d = new Date(iso);
    return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
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

export default function HistoryScreen() {
  const { user } = useAuth();
  const { width: screenWidth } = useWindowDimensions();
  const isMobile = screenWidth < MOBILE_BREAKPOINT;
  
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [source, setSource] = useState('local'); // 'local' or 'cloud'

  const refresh = useCallback(async () => {
    setLoading(true);
    
    if (user) {
      // User logged in - fetch from MongoDB
      try {
        const result = await historyService.listHistory({ limit: 50 });
        const cloudItems = result.items.map(item => ({
          id: item.id,
          createdAt: item.createdAt,
          imageName: item.imageName,
          category: item.category,
          confidence: item.confidence,
          ratio: (item.oilYieldPercent || 0) / 100,
          oilYieldPercent: item.oilYieldPercent,
          yieldCategory: item.yieldCategory,
          referenceDetected: item.referenceDetected,
          inputs: {
            lengthMm: item.dimensions?.lengthCm ? item.dimensions.lengthCm * 10 : null,
            widthMm: item.dimensions?.widthCm ? item.dimensions.widthCm * 10 : null,
            weightG: item.dimensions?.wholeFruitWeightG ?? null,
          },
        }));
        setItems(cloudItems);
        setSource('cloud');
      } catch (e) {
        console.warn('[HistoryScreen] Cloud fetch failed, using local', e);
        const local = await loadLocalHistory();
        setItems(local);
        setSource('local');
      }
    } else {
      // Not logged in - use local storage
      const local = await loadLocalHistory();
      setItems(local);
      setSource('local');
    }
    
    setLoading(false);
  }, [user]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function onClear() {
    if (Platform.OS === 'web') {
      if (!window.confirm('Clear all history?')) return;
    } else {
      const ok = await new Promise((resolve) => {
        Alert.alert('Clear History', 'Clear all saved predictions?', [
          { text: 'Cancel', style: 'cancel', onPress: () => resolve(false) },
          { text: 'Clear', style: 'destructive', onPress: () => resolve(true) },
        ]);
      });
      if (!ok) return;
    }
    
    if (user) {
      await historyService.clearAllHistory();
    }
    await clearLocalHistory();
    await refresh();
  }

  return (
    <Screen scroll={false}>
      {/* Compact Header */}
      <View style={[styles.header, isMobile && styles.headerMobile]}>
        <View style={styles.headerLeft}>
          <Ionicons name="time" size={isMobile ? 22 : 20} color="#fff" />
          <Text style={[styles.headerTitle, isMobile && styles.headerTitleMobile]}>History</Text>
          <View style={[styles.sourceBadge, source === 'cloud' && styles.sourceBadgeCloud]}>
            <Ionicons name={source === 'cloud' ? 'cloud' : 'phone-portrait'} size={isMobile ? 12 : 10} color="#fff" />
            <Text style={[styles.sourceText, isMobile && styles.sourceTextMobile]}>{source === 'cloud' ? 'Cloud' : 'Local'}</Text>
          </View>
        </View>
        <View style={styles.headerActions}>
          <Pressable onPress={refresh} style={[styles.headerBtn, isMobile && styles.headerBtnMobile]}>
            <Ionicons name="refresh" size={isMobile ? 20 : 16} color="#fff" />
          </Pressable>
          <Pressable onPress={onClear} style={[styles.headerBtn, isMobile && styles.headerBtnMobile, { backgroundColor: 'rgba(244,67,54,0.3)' }]}>
            <Ionicons name="trash" size={isMobile ? 20 : 16} color="#fff" />
          </Pressable>
        </View>
      </View>

      {/* Stats Row */}
      {!loading && items.length > 0 && (
        <View style={[styles.statsRow, isMobile && styles.statsRowMobile]}>
          <View style={[styles.statBox, isMobile && styles.statBoxMobile]}>
            <Text style={[styles.statNum, isMobile && styles.statNumMobile]}>{items.length}</Text>
            <Text style={[styles.statLabel, isMobile && styles.statLabelMobile]}>Total</Text>
          </View>
          <View style={[styles.statBox, isMobile && styles.statBoxMobile, { backgroundColor: '#e8f5e9' }]}>
            <Text style={[styles.statNum, isMobile && styles.statNumMobile, { color: '#4caf50' }]}>{items.filter(i => i.category === 'GREEN').length}</Text>
            <Text style={[styles.statLabel, isMobile && styles.statLabelMobile]}>Green</Text>
          </View>
          <View style={[styles.statBox, isMobile && styles.statBoxMobile, { backgroundColor: '#fff8e1' }]}>
            <Text style={[styles.statNum, isMobile && styles.statNumMobile, { color: '#f5a623' }]}>{items.filter(i => i.category === 'YELLOW').length}</Text>
            <Text style={[styles.statLabel, isMobile && styles.statLabelMobile]}>Yellow</Text>
          </View>
          <View style={[styles.statBox, isMobile && styles.statBoxMobile, { backgroundColor: '#efebe9' }]}>
            <Text style={[styles.statNum, isMobile && styles.statNumMobile, { color: '#8b4513' }]}>{items.filter(i => i.category === 'BROWN').length}</Text>
            <Text style={[styles.statLabel, isMobile && styles.statLabelMobile]}>Brown</Text>
          </View>
        </View>
      )}

      {/* List */}
      {loading ? (
        <View style={styles.emptyState}>
          <Ionicons name="hourglass" size={isMobile ? 48 : 32} color="#ccc" />
          <Text style={[styles.emptyText, isMobile && styles.emptyTextMobile]}>Loading...</Text>
        </View>
      ) : items.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="folder-open-outline" size={isMobile ? 48 : 32} color="#ccc" />
          <Text style={[styles.emptyText, isMobile && styles.emptyTextMobile]}>No saved results</Text>
          <Text style={[styles.emptyHint, isMobile && styles.emptyHintMobile]}>Scan a fruit and save the result</Text>
        </View>
      ) : (
        <FlatList
          data={items}
          keyExtractor={(item) => item.id}
          contentContainerStyle={{ paddingBottom: isMobile ? 24 : 16 }}
          renderItem={({ item }) => (
            <View style={[styles.card, isMobile && styles.cardMobile]}>
              <View style={[styles.colorBar, isMobile && styles.colorBarMobile, { backgroundColor: getCategoryColor(item.category) }]} />
              <View style={[styles.cardContent, isMobile && styles.cardContentMobile]}>
                <View style={styles.cardTop}>
                  <View style={[styles.catBadge, isMobile && styles.catBadgeMobile, { backgroundColor: getCategoryColor(item.category) }]}>
                    <Text style={[styles.catText, isMobile && styles.catTextMobile]}>{item.category}</Text>
                  </View>
                  <Text style={[styles.dateText, isMobile && styles.dateTextMobile]}>{formatDate(item.createdAt)}</Text>
                </View>
                {item.imageName && (
                  <View style={[styles.filenameRow, isMobile && styles.filenameRowMobile]}>
                    <Ionicons name="document" size={isMobile ? 14 : 10} color="#666" />
                    <Text style={[styles.filenameText, isMobile && styles.filenameTextMobile]} numberOfLines={1}>{item.imageName}</Text>
                  </View>
                )}
                <View style={styles.cardBottom}>
                  <Text style={[styles.ratioNum, isMobile && styles.ratioNumMobile]}>{Math.round(item.ratio * 100)}%</Text>
                  <View style={[styles.dims, isMobile && styles.dimsMobile]}>
                    <Text style={[styles.dimText, isMobile && styles.dimTextMobile]}>L: {item.inputs?.lengthMm?.toFixed(0) ?? '—'}mm</Text>
                    <Text style={[styles.dimText, isMobile && styles.dimTextMobile]}>W: {item.inputs?.widthMm?.toFixed(0) ?? '—'}mm</Text>
                    <Text style={[styles.dimText, isMobile && styles.dimTextMobile]}>Wt: {item.inputs?.weightG?.toFixed(0) ?? '—'}g</Text>
                  </View>
                </View>
              </View>
            </View>
          )}
        />
      )}
    </Screen>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: theme.colors.greenDark,
    padding: 10,
    borderRadius: 8,
    marginBottom: 8,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  headerTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  sourceBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
    backgroundColor: 'rgba(255,255,255,0.2)',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 8,
  },
  sourceBadgeCloud: {
    backgroundColor: '#2196f3',
  },
  sourceText: {
    color: '#fff',
    fontSize: 9,
    fontWeight: '600',
  },
  headerActions: {
    flexDirection: 'row',
    gap: 6,
  },
  headerBtn: {
    padding: 6,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 6,
  },

  statsRow: {
    flexDirection: 'row',
    gap: 6,
    marginBottom: 8,
  },
  statBox: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 6,
    paddingVertical: 8,
    alignItems: 'center',
  },
  statNum: {
    fontSize: 18,
    fontWeight: '800',
    color: theme.colors.greenDark,
  },
  statLabel: {
    fontSize: 9,
    color: '#666',
    marginTop: 1,
  },

  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  emptyText: {
    color: '#999',
    fontSize: 14,
  },
  emptyHint: {
    color: '#bbb',
    fontSize: 11,
  },

  card: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 8,
    marginBottom: 6,
    overflow: 'hidden',
  },
  colorBar: {
    width: 4,
  },
  cardContent: {
    flex: 1,
    padding: 10,
  },
  cardTop: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  catBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  catText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '700',
  },
  dateText: {
    color: '#999',
    fontSize: 10,
  },
  filenameRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginBottom: 4,
  },
  filenameText: {
    flex: 1,
    color: '#666',
    fontSize: 9,
  },
  cardBottom: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  ratioNum: {
    fontSize: 24,
    fontWeight: '900',
    color: theme.colors.greenDark,
  },
  dims: {
    flexDirection: 'row',
    gap: 8,
  },
  dimText: {
    fontSize: 10,
    color: '#666',
  },

  /* ========== MOBILE RESPONSIVE STYLES ========== */
  
  /* Header - Mobile */
  headerMobile: {
    padding: 14,
    borderRadius: 12,
    marginBottom: 12,
  },
  headerTitleMobile: {
    fontSize: 18,
  },
  sourceTextMobile: {
    fontSize: 11,
  },
  headerBtnMobile: {
    padding: 10,
    borderRadius: 8,
  },

  /* Stats Row - Mobile */
  statsRowMobile: {
    gap: 10,
    marginBottom: 12,
  },
  statBoxMobile: {
    borderRadius: 10,
    paddingVertical: 12,
  },
  statNumMobile: {
    fontSize: 24,
  },
  statLabelMobile: {
    fontSize: 12,
    marginTop: 2,
  },

  /* Empty State - Mobile */
  emptyTextMobile: {
    fontSize: 18,
  },
  emptyHintMobile: {
    fontSize: 14,
  },

  /* Card - Mobile */
  cardMobile: {
    borderRadius: 12,
    marginBottom: 10,
  },
  colorBarMobile: {
    width: 6,
  },
  cardContentMobile: {
    padding: 14,
  },
  catBadgeMobile: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  catTextMobile: {
    fontSize: 14,
    fontWeight: '800',
  },
  dateTextMobile: {
    fontSize: 13,
  },
  filenameRowMobile: {
    marginBottom: 8,
    gap: 6,
  },
  filenameTextMobile: {
    fontSize: 13,
  },
  ratioNumMobile: {
    fontSize: 32,
  },
  dimsMobile: {
    gap: 12,
  },
  dimTextMobile: {
    fontSize: 13,
  },
});
