import React, { useEffect, useState } from 'react';
import { Dimensions, Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Screen from '../components/Screen';
import { useAuth } from '../hooks';
import { adminService } from '../services';
import { theme } from '../theme/theme';

const screenWidth = Dimensions.get('window').width;

// Simple bar chart component
function BarChart({ data, title, color = theme.colors.green, maxBars = 7 }) {
  if (!data || data.length === 0) return null;
  
  const chartData = data.slice(-maxBars);
  const maxValue = Math.max(...chartData.map(d => d.count || d.avgYield || 0), 1);
  
  return (
    <View style={styles.chartContainer}>
      <Text style={styles.chartTitle}>{title}</Text>
      <View style={styles.barChartWrap}>
        {chartData.map((item, index) => {
          const value = item.count || item.avgYield || 0;
          const height = Math.max((value / maxValue) * 80, 4);
          return (
            <View key={index} style={styles.barColumn}>
              <Text style={styles.barValue}>{Math.round(value)}</Text>
              <View style={[styles.bar, { height, backgroundColor: color }]} />
              <Text style={styles.barLabel} numberOfLines={1}>
                {item._id?.split('-').pop() || item._id || ''}
              </Text>
            </View>
          );
        })}
      </View>
    </View>
  );
}

// Pie chart simulation with segments
function PieChartSimple({ data, title }) {
  const total = Object.values(data).reduce((a, b) => a + b, 0);
  if (total === 0) return null;
  
  const colors = { GREEN: '#4caf50', YELLOW: '#ffc107', BROWN: '#8b4513' };
  
  return (
    <View style={styles.chartContainer}>
      <Text style={styles.chartTitle}>{title}</Text>
      <View style={styles.pieWrap}>
        <View style={styles.pieVisual}>
          {Object.entries(data).map(([key, value], index) => {
            const percent = Math.round((value / total) * 100);
            return (
              <View 
                key={key} 
                style={[
                  styles.pieSegment, 
                  { 
                    backgroundColor: colors[key] || '#999',
                    flex: value,
                  }
                ]} 
              />
            );
          })}
        </View>
        <View style={styles.pieLegend}>
          {Object.entries(data).map(([key, value]) => (
            <View key={key} style={styles.legendItem}>
              <View style={[styles.legendDot, { backgroundColor: colors[key] || '#999' }]} />
              <Text style={styles.legendText}>{key}: {value} ({Math.round((value / total) * 100)}%)</Text>
            </View>
          ))}
        </View>
      </View>
    </View>
  );
}

export default function AdminDashboardScreen({ navigation }) {
  const { isAdmin } = useAuth();
  const [analytics, setAnalytics] = useState(null);
  const [historyItems, setHistoryItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview'); // overview, history, users, charts

  async function loadAnalytics() {
    setLoading(true);
    setError(null);
    try {
      const [data, history] = await Promise.all([
        adminService.getAnalyticsOverview(),
        adminService.listAllHistory(50),
      ]);
      setAnalytics(data);
      setHistoryItems(history);
    } catch (e) {
      console.warn('[AdminDashboard]', e?.message);
      setError('Failed to load analytics');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (isAdmin) loadAnalytics();
  }, [isAdmin]);

  if (!isAdmin) {
    return (
      <Screen>
        <View style={styles.accessDenied}>
          <Ionicons name="lock-closed" size={48} color={theme.colors.danger} />
          <Text style={styles.deniedTitle}>Access Denied</Text>
          <Text style={styles.deniedText}>Admin access required.</Text>
          <Pressable style={styles.backBtn} onPress={() => navigation.goBack()}>
            <Text style={styles.backBtnText}>Go Back</Text>
          </Pressable>
        </View>
      </Screen>
    );
  }

  return (
    <Screen scroll={false}>
      {/* Header */}
      <View style={styles.header}>
        <Pressable onPress={() => navigation.goBack()} style={styles.headerBackBtn}>
          <Ionicons name="arrow-back" size={20} color="#fff" />
        </Pressable>
        <View style={styles.headerContent}>
          <Ionicons name="analytics" size={22} color="#fff" />
          <Text style={styles.headerTitle}>Analytics Dashboard</Text>
        </View>
        <Pressable onPress={loadAnalytics} style={styles.headerRefreshBtn} disabled={loading}>
          <Ionicons name={loading ? "hourglass" : "refresh"} size={18} color="#fff" />
        </Pressable>
      </View>

      {/* Tab Navigation */}
      <View style={styles.tabRow}>
        {[
          { id: 'overview', label: 'Overview', icon: 'grid' },
          { id: 'history', label: 'History', icon: 'time' },
          { id: 'users', label: 'Users', icon: 'people' },
          { id: 'charts', label: 'Charts', icon: 'bar-chart' },
        ].map(tab => (
          <Pressable 
            key={tab.id}
            style={[styles.tab, activeTab === tab.id && styles.tabActive]}
            onPress={() => setActiveTab(tab.id)}
          >
            <Ionicons name={tab.icon} size={14} color={activeTab === tab.id ? '#fff' : '#666'} />
            <Text style={[styles.tabText, activeTab === tab.id && styles.tabTextActive]}>{tab.label}</Text>
          </Pressable>
        ))}
      </View>

      {error && (
        <View style={styles.errorBox}>
          <Ionicons name="warning" size={16} color="#f44336" />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      {loading ? (
        <View style={styles.loadingBox}>
          <Ionicons name="hourglass" size={32} color="#ccc" />
          <Text style={styles.loadingText}>Loading analytics...</Text>
        </View>
      ) : (
        <ScrollView style={styles.content} contentContainerStyle={{ paddingBottom: 24 }}>
          {/* Overview Tab */}
          {activeTab === 'overview' && analytics && (
            <>
              {/* Top Stats Row - 4 compact cards */}
              <View style={styles.topStatsRow}>
                <View style={[styles.topStatCard, { borderLeftColor: '#1976d2' }]}>
                  <Ionicons name="people" size={18} color="#1976d2" />
                  <View style={styles.topStatContent}>
                    <Text style={styles.topStatValue}>{analytics.overview?.totalUsers || 0}</Text>
                    <Text style={styles.topStatLabel}>Users</Text>
                  </View>
                </View>
                <View style={[styles.topStatCard, { borderLeftColor: '#388e3c' }]}>
                  <Ionicons name="scan" size={18} color="#388e3c" />
                  <View style={styles.topStatContent}>
                    <Text style={styles.topStatValue}>{analytics.overview?.totalHistory || 0}</Text>
                    <Text style={styles.topStatLabel}>Scans</Text>
                  </View>
                </View>
                <View style={[styles.topStatCard, { borderLeftColor: '#f57c00' }]}>
                  <Ionicons name="water" size={18} color="#f57c00" />
                  <View style={styles.topStatContent}>
                    <Text style={styles.topStatValue}>{Math.round(analytics.avgYieldOverall || 0)}%</Text>
                    <Text style={styles.topStatLabel}>Avg Yield</Text>
                  </View>
                </View>
                <View style={[styles.topStatCard, { borderLeftColor: '#c2185b' }]}>
                  <Ionicons name="disc" size={18} color="#c2185b" />
                  <View style={styles.topStatContent}>
                    <Text style={styles.topStatValue}>
                      {analytics.coinDetection?.totalScans > 0 
                        ? Math.round((analytics.coinDetection.withCoin / analytics.coinDetection.totalScans) * 100) 
                        : 0}%
                    </Text>
                    <Text style={styles.topStatLabel}>Coin Rate</Text>
                  </View>
                </View>
              </View>

              {/* Two Column Layout */}
              <View style={styles.twoColumnRow}>
                {/* Left Column */}
                <View style={styles.overviewColumn}>
                  {/* Category Distribution */}
                  <PieChartSimple 
                    data={analytics.categoryDistribution || {}} 
                    title="Category Distribution" 
                  />

                  {/* Average Yield by Category */}
                  <View style={styles.cardCompact}>
                    <Text style={styles.cardTitleSmall}>Oil Yield by Category</Text>
                    {Object.entries(analytics.avgYieldByCategory || {}).map(([cat, data]) => (
                      <View key={cat} style={styles.yieldRowCompact}>
                        <View style={[styles.yieldDotSmall, { 
                          backgroundColor: cat === 'GREEN' ? '#4caf50' : cat === 'YELLOW' ? '#ffc107' : '#8b4513' 
                        }]} />
                        <Text style={styles.yieldCatSmall}>{cat}</Text>
                        <View style={styles.yieldBarWrapSmall}>
                          <View style={[styles.yieldBar, { 
                            width: `${data.avg}%`,
                            backgroundColor: cat === 'GREEN' ? '#4caf50' : cat === 'YELLOW' ? '#ffc107' : '#8b4513'
                          }]} />
                        </View>
                        <Text style={styles.yieldValueSmall}>{data.avg}%</Text>
                      </View>
                    ))}
                  </View>
                </View>

                {/* Right Column */}
                <View style={styles.overviewColumn}>
                  {/* Confidence Stats - Compact */}
                  <View style={styles.cardCompact}>
                    <Text style={styles.cardTitleSmall}>Prediction Confidence</Text>
                    <View style={styles.confidenceRow}>
                      <View style={styles.confItem}>
                        <Text style={styles.confValue}>{Math.round((analytics.confidenceStats?.avgConfidence || 0) * 100)}%</Text>
                        <Text style={styles.confLabel}>Avg</Text>
                      </View>
                      <View style={styles.confItem}>
                        <Text style={[styles.confValue, { color: '#f44336' }]}>{Math.round((analytics.confidenceStats?.minConfidence || 0) * 100)}%</Text>
                        <Text style={styles.confLabel}>Min</Text>
                      </View>
                      <View style={styles.confItem}>
                        <Text style={[styles.confValue, { color: '#4caf50' }]}>{Math.round((analytics.confidenceStats?.maxConfidence || 0) * 100)}%</Text>
                        <Text style={styles.confLabel}>Max</Text>
                      </View>
                    </View>
                  </View>

                  {/* Quick Info Cards */}
                  <View style={styles.quickInfoRow}>
                    <View style={[styles.quickInfoCard, { backgroundColor: '#e8f5e9' }]}>
                      <Text style={styles.quickInfoValue}>
                        +{analytics.overview?.newUsersThisMonth || 0}
                      </Text>
                      <Text style={styles.quickInfoLabel}>New Users</Text>
                    </View>
                    <View style={[styles.quickInfoCard, { backgroundColor: '#fff3e0' }]}>
                      <Text style={styles.quickInfoValue}>
                        {analytics.spotStats?.withSpots || 0}
                      </Text>
                      <Text style={styles.quickInfoLabel}>With Spots</Text>
                    </View>
                  </View>

                  {/* Dimension Stats - Compact */}
                  <View style={styles.cardCompact}>
                    <Text style={styles.cardTitleSmall}>Dimensions by Category</Text>
                    {Object.entries(analytics.dimensionStats || {}).map(([cat, data]) => (
                      <View key={cat} style={styles.dimRowCompact}>
                        <Text style={[styles.dimCatCompact, { 
                          color: cat === 'GREEN' ? '#4caf50' : cat === 'YELLOW' ? '#f5a623' : '#8b4513' 
                        }]}>{cat}</Text>
                        <Text style={styles.dimTextCompact}>{data.avgLength}×{data.avgWidth}cm</Text>
                        <Text style={styles.dimTextCompact}>{data.avgWeight}g</Text>
                        <Text style={styles.dimCountCompact}>({data.count})</Text>
                      </View>
                    ))}
                  </View>
                </View>
              </View>
            </>
          )}

          {/* Users Tab */}
          {activeTab === 'users' && analytics && (
            <>
              <View style={styles.card}>
                <Text style={styles.cardTitle}>Top Active Users</Text>
                {(analytics.userActivity || []).length === 0 ? (
                  <Text style={styles.emptyText}>No user activity yet</Text>
                ) : (
                  analytics.userActivity.map((user, index) => (
                    <View key={user.userId} style={styles.userRow}>
                      <View style={styles.userRank}>
                        <Text style={styles.userRankText}>#{index + 1}</Text>
                      </View>
                      <View style={styles.userInfo}>
                        <Text style={styles.userEmail} numberOfLines={1}>{user.email}</Text>
                        <Text style={styles.userMeta}>
                          {user.scanCount} scans • Last: {new Date(user.lastScan).toLocaleDateString()}
                        </Text>
                      </View>
                      <View style={styles.userScans}>
                        <Ionicons name="scan" size={14} color={theme.colors.green} />
                        <Text style={styles.userScansText}>{user.scanCount}</Text>
                      </View>
                    </View>
                  ))
                )}
              </View>

              {/* User Stats Summary */}
              <View style={styles.card}>
                <Text style={styles.cardTitle}>User Statistics</Text>
                <View style={styles.userStatsGrid}>
                  <View style={styles.userStatBox}>
                    <Ionicons name="people" size={20} color={theme.colors.green} />
                    <Text style={styles.userStatValue}>{analytics.overview?.totalUsers || 0}</Text>
                    <Text style={styles.userStatLabel}>Total Users</Text>
                  </View>
                  <View style={styles.userStatBox}>
                    <Ionicons name="person-add" size={20} color="#2196f3" />
                    <Text style={styles.userStatValue}>{analytics.overview?.newUsersThisMonth || 0}</Text>
                    <Text style={styles.userStatLabel}>New This Month</Text>
                  </View>
                  <View style={styles.userStatBox}>
                    <Ionicons name="trending-up" size={20} color="#ff9800" />
                    <Text style={styles.userStatValue}>
                      {analytics.overview?.totalHistory && analytics.overview?.totalUsers
                        ? (analytics.overview.totalHistory / analytics.overview.totalUsers).toFixed(1)
                        : '0'}
                    </Text>
                    <Text style={styles.userStatLabel}>Avg Scans/User</Text>
                  </View>
                </View>
              </View>
            </>
          )}

          {/* History Tab - Scan/Prediction History */}
          {activeTab === 'history' && (
            <>
              <View style={styles.card}>
                <View style={styles.cardTitleRow}>
                  <Text style={styles.cardTitle}>Recent Scan History</Text>
                  <Text style={styles.cardCount}>{historyItems.length} records</Text>
                </View>
                
                {historyItems.length === 0 ? (
                  <View style={styles.emptyState}>
                    <Ionicons name="folder-open" size={32} color="#ccc" />
                    <Text style={styles.emptyText}>No scan history yet</Text>
                  </View>
                ) : (
                  historyItems.map((item, index) => (
                    <View key={item.id || index} style={styles.historyCard}>
                      {/* Header with user and date */}
                      <View style={styles.historyHeader}>
                        <View style={styles.historyUser}>
                          <Ionicons name="person" size={12} color="#666" />
                          <Text style={styles.historyUserText} numberOfLines={1}>
                            {item.userEmail || 'Unknown'}
                          </Text>
                        </View>
                        <Text style={styles.historyDate}>
                          {new Date(item.createdAt).toLocaleDateString()} {new Date(item.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </Text>
                      </View>

                      {/* Main content row */}
                      <View style={styles.historyContent}>
                        {/* Image thumbnail */}
                        {item.imageUri ? (
                          <Image 
                            source={{ uri: item.imageUri }} 
                            style={styles.historyThumb}
                            resizeMode="cover"
                          />
                        ) : (
                          <View style={[styles.historyThumb, styles.historyThumbPlaceholder]}>
                            <Ionicons name="leaf" size={20} color="#ccc" />
                          </View>
                        )}

                        {/* Details */}
                        <View style={styles.historyDetails}>
                          {/* Category and yield */}
                          <View style={styles.historyMainRow}>
                            <View style={[styles.historyCatBadge, { 
                              backgroundColor: item.category === 'GREEN' ? '#4caf50' 
                                : item.category === 'YELLOW' ? '#ffc107' : '#8b4513' 
                            }]}>
                              <Text style={styles.historyCatText}>{item.category}</Text>
                            </View>
                            <Text style={styles.historyYield}>{item.oilYieldPercent?.toFixed(1) || '—'}%</Text>
                            <Text style={styles.historyYieldLabel}>oil yield</Text>
                          </View>

                          {/* Dimensions */}
                          <View style={styles.historyDimRow}>
                            <Text style={styles.historyDimText}>
                              L: {item.dimensions?.lengthCm?.toFixed(1) || '—'} cm
                            </Text>
                            <Text style={styles.historyDimText}>
                              W: {item.dimensions?.widthCm?.toFixed(1) || '—'} cm
                            </Text>
                            <Text style={styles.historyDimText}>
                              Wt: {item.dimensions?.wholeFruitWeightG?.toFixed(0) || '—'} g
                            </Text>
                          </View>

                          {/* Confidence and coin status */}
                          <View style={styles.historyMetaRow}>
                            <View style={styles.historyMetaItem}>
                              <Ionicons name="speedometer" size={10} color="#666" />
                              <Text style={styles.historyMetaText}>
                                {Math.round((item.oilConfidence || item.confidence || 0) * 100)}% conf
                              </Text>
                            </View>
                            <View style={styles.historyMetaItem}>
                              <Ionicons 
                                name={item.referenceDetected ? "checkmark-circle" : "alert-circle"} 
                                size={10} 
                                color={item.referenceDetected ? "#4caf50" : "#ff9800"} 
                              />
                              <Text style={styles.historyMetaText}>
                                {item.referenceDetected ? 'Coin' : 'No coin'}
                              </Text>
                            </View>
                            {item.hasSpots && (
                              <View style={styles.historyMetaItem}>
                                <Ionicons name="alert" size={10} color="#ff9800" />
                                <Text style={styles.historyMetaText}>
                                  Spots {item.spotCoverage?.toFixed(0) || 0}%
                                </Text>
                              </View>
                            )}
                          </View>

                          {/* Yield category */}
                          <Text style={styles.historyYieldCat}>
                            {item.yieldCategory || 'Unknown'} • {item.maturityStage || 'Unknown'}
                          </Text>
                        </View>
                      </View>

                      {/* Interpretation */}
                      {item.interpretation && (
                        <Text style={styles.historyInterp} numberOfLines={2}>
                          {item.interpretation}
                        </Text>
                      )}
                    </View>
                  ))
                )}
              </View>
            </>
          )}

          {/* Charts Tab */}
          {activeTab === 'charts' && analytics && (
            <>
              {/* Daily Activity */}
              <BarChart 
                data={analytics.dailyActivity || []} 
                title="Daily Scan Activity (Last 7 Days)"
                color={theme.colors.green}
              />

              {/* Weekly Trend */}
              <BarChart 
                data={analytics.weeklyTrend || []} 
                title="Weekly Trend (Avg Oil Yield %)"
                color="#ff9800"
              />

              {/* Yield Distribution */}
              <View style={styles.card}>
                <Text style={styles.cardTitle}>Yield Category Distribution</Text>
                {Object.entries(analytics.yieldDistribution || {}).length === 0 ? (
                  <Text style={styles.emptyText}>No data available</Text>
                ) : (
                  Object.entries(analytics.yieldDistribution).map(([cat, count]) => {
                    const total = Object.values(analytics.yieldDistribution).reduce((a, b) => a + b, 0);
                    const percent = Math.round((count / total) * 100);
                    return (
                      <View key={cat} style={styles.yieldDistRow}>
                        <Text style={styles.yieldDistCat}>{cat}</Text>
                        <View style={styles.yieldDistBarWrap}>
                          <View style={[styles.yieldDistBar, { width: `${percent}%` }]} />
                        </View>
                        <Text style={styles.yieldDistValue}>{count} ({percent}%)</Text>
                      </View>
                    );
                  })
                )}
              </View>
            </>
          )}
        </ScrollView>
      )}
    </Screen>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.greenDark,
    padding: 10,
    borderRadius: 8,
    marginBottom: 8,
  },
  headerBackBtn: {
    padding: 6,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 6,
    marginRight: 10,
  },
  headerContent: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  headerTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  headerRefreshBtn: {
    padding: 6,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 6,
  },

  tabRow: {
    flexDirection: 'row',
    gap: 6,
    marginBottom: 8,
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    paddingVertical: 8,
    backgroundColor: '#fff',
    borderRadius: 6,
  },
  tabActive: {
    backgroundColor: theme.colors.green,
  },
  tabText: {
    fontSize: 11,
    color: '#666',
    fontWeight: '600',
  },
  tabTextActive: {
    color: '#fff',
  },

  errorBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#ffebee',
    padding: 10,
    borderRadius: 6,
    marginBottom: 8,
  },
  errorText: {
    color: '#c62828',
    fontSize: 12,
  },

  loadingBox: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  },
  loadingText: {
    color: '#999',
    fontSize: 13,
  },

  content: {
    flex: 1,
  },

  /* New Compact Overview Styles */
  topStatsRow: {
    flexDirection: 'row',
    gap: 6,
    marginBottom: 8,
  },
  topStatCard: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#fff',
    padding: 10,
    borderRadius: 6,
    borderLeftWidth: 3,
  },
  topStatContent: {
    flex: 1,
  },
  topStatValue: {
    fontSize: 18,
    fontWeight: '800',
    color: '#333',
  },
  topStatLabel: {
    fontSize: 9,
    color: '#666',
  },

  twoColumnRow: {
    flexDirection: 'row',
    gap: 8,
  },
  overviewColumn: {
    flex: 1,
    gap: 8,
  },

  cardCompact: {
    backgroundColor: '#fff',
    borderRadius: 6,
    padding: 10,
  },
  cardTitleSmall: {
    fontSize: 11,
    fontWeight: '700',
    color: '#333',
    marginBottom: 8,
  },

  yieldRowCompact: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginBottom: 4,
  },
  yieldDotSmall: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  yieldCatSmall: {
    width: 40,
    fontSize: 9,
    fontWeight: '600',
    color: '#333',
  },
  yieldBarWrapSmall: {
    flex: 1,
    height: 6,
    backgroundColor: '#eee',
    borderRadius: 3,
    overflow: 'hidden',
  },
  yieldValueSmall: {
    width: 30,
    fontSize: 9,
    fontWeight: '700',
    color: '#333',
    textAlign: 'right',
  },

  confidenceRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  confItem: {
    alignItems: 'center',
  },
  confValue: {
    fontSize: 16,
    fontWeight: '800',
    color: theme.colors.greenDark,
  },
  confLabel: {
    fontSize: 8,
    color: '#666',
  },

  quickInfoRow: {
    flexDirection: 'row',
    gap: 6,
  },
  quickInfoCard: {
    flex: 1,
    padding: 8,
    borderRadius: 6,
    alignItems: 'center',
  },
  quickInfoValue: {
    fontSize: 14,
    fontWeight: '800',
    color: '#333',
  },
  quickInfoLabel: {
    fontSize: 8,
    color: '#666',
  },

  dimRowCompact: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 3,
  },
  dimCatCompact: {
    width: 45,
    fontSize: 9,
    fontWeight: '700',
  },
  dimTextCompact: {
    fontSize: 9,
    color: '#666',
  },
  dimCountCompact: {
    fontSize: 8,
    color: '#999',
  },

  /* Original styles for other tabs */
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginBottom: 8,
  },
  metricCard: {
    width: (screenWidth - 50) / 2,
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  metricValue: {
    fontSize: 28,
    fontWeight: '900',
    color: '#333',
    marginTop: 4,
  },
  metricLabel: {
    fontSize: 11,
    color: '#666',
    fontWeight: '600',
  },
  metricSub: {
    fontSize: 9,
    color: '#999',
    marginTop: 2,
  },

  chartContainer: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  chartTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#333',
    marginBottom: 12,
  },
  barChartWrap: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-around',
    height: 120,
  },
  barColumn: {
    alignItems: 'center',
    flex: 1,
  },
  barValue: {
    fontSize: 9,
    color: '#666',
    marginBottom: 2,
  },
  bar: {
    width: 20,
    borderRadius: 3,
  },
  barLabel: {
    fontSize: 8,
    color: '#999',
    marginTop: 4,
    width: 30,
    textAlign: 'center',
  },

  pieWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  pieVisual: {
    flexDirection: 'row',
    width: 120,
    height: 20,
    borderRadius: 10,
    overflow: 'hidden',
  },
  pieSegment: {
    height: '100%',
  },
  pieLegend: {
    flex: 1,
    gap: 4,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  legendDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  legendText: {
    fontSize: 11,
    color: '#666',
  },

  card: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  cardTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#333',
    marginBottom: 10,
  },

  yieldRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  yieldDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  yieldCat: {
    width: 50,
    fontSize: 11,
    fontWeight: '600',
    color: '#333',
  },
  yieldBarWrap: {
    flex: 1,
    height: 8,
    backgroundColor: '#eee',
    borderRadius: 4,
    overflow: 'hidden',
  },
  yieldBar: {
    height: '100%',
    borderRadius: 4,
  },
  yieldValue: {
    width: 40,
    fontSize: 11,
    fontWeight: '700',
    color: '#333',
    textAlign: 'right',
  },

  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
    gap: 2,
  },
  statValue: {
    fontSize: 24,
    fontWeight: '800',
    color: theme.colors.greenDark,
  },
  statLabel: {
    fontSize: 10,
    color: '#666',
  },

  quickStatsRow: {
    flexDirection: 'row',
    gap: 6,
    marginBottom: 8,
  },
  quickStat: {
    flex: 1,
    padding: 10,
    borderRadius: 8,
    alignItems: 'center',
    gap: 4,
  },
  quickStatValue: {
    fontSize: 16,
    fontWeight: '800',
    color: '#333',
  },
  quickStatLabel: {
    fontSize: 9,
    color: '#666',
  },

  dimRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  dimCat: {
    width: 50,
    fontSize: 11,
    fontWeight: '700',
  },
  dimText: {
    fontSize: 10,
    color: '#666',
  },
  dimCount: {
    fontSize: 9,
    color: '#999',
  },

  userRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  userRank: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: theme.colors.green,
    alignItems: 'center',
    justifyContent: 'center',
  },
  userRankText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '700',
  },
  userInfo: {
    flex: 1,
  },
  userEmail: {
    fontSize: 12,
    fontWeight: '600',
    color: '#333',
  },
  userMeta: {
    fontSize: 10,
    color: '#999',
  },
  userScans: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  userScansText: {
    fontSize: 12,
    fontWeight: '700',
    color: theme.colors.green,
  },

  userStatsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  userStatBox: {
    alignItems: 'center',
    padding: 10,
  },
  userStatValue: {
    fontSize: 20,
    fontWeight: '800',
    color: '#333',
    marginTop: 4,
  },
  userStatLabel: {
    fontSize: 9,
    color: '#666',
  },

  yieldDistRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  yieldDistCat: {
    width: 80,
    fontSize: 10,
    color: '#333',
  },
  yieldDistBarWrap: {
    flex: 1,
    height: 12,
    backgroundColor: '#e8f5e9',
    borderRadius: 6,
    overflow: 'hidden',
  },
  yieldDistBar: {
    height: '100%',
    backgroundColor: theme.colors.green,
    borderRadius: 6,
  },
  yieldDistValue: {
    width: 60,
    fontSize: 10,
    color: '#666',
    textAlign: 'right',
  },

  emptyText: {
    color: '#999',
    fontSize: 12,
    textAlign: 'center',
    paddingVertical: 16,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 24,
    gap: 8,
  },

  /* History Tab Styles */
  cardTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  cardCount: {
    fontSize: 10,
    color: '#999',
    backgroundColor: '#f0f0f0',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  historyCard: {
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    padding: 10,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  historyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
    paddingBottom: 6,
    borderBottomWidth: 1,
    borderBottomColor: '#e8e8e8',
  },
  historyUser: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    flex: 1,
  },
  historyUserText: {
    fontSize: 10,
    color: '#666',
    flex: 1,
  },
  historyDate: {
    fontSize: 9,
    color: '#999',
  },
  historyContent: {
    flexDirection: 'row',
    gap: 10,
  },
  historyThumb: {
    width: 60,
    height: 60,
    borderRadius: 6,
    backgroundColor: '#f0f0f0',
  },
  historyThumbPlaceholder: {
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  historyDetails: {
    flex: 1,
  },
  historyMainRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
  },
  historyCatBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  historyCatText: {
    color: '#fff',
    fontSize: 9,
    fontWeight: '700',
  },
  historyYield: {
    fontSize: 18,
    fontWeight: '800',
    color: theme.colors.greenDark,
  },
  historyYieldLabel: {
    fontSize: 9,
    color: '#999',
  },
  historyDimRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 4,
  },
  historyDimText: {
    fontSize: 9,
    color: '#666',
  },
  historyMetaRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 2,
  },
  historyMetaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
  },
  historyMetaText: {
    fontSize: 8,
    color: '#888',
  },
  historyYieldCat: {
    fontSize: 9,
    color: '#666',
    fontStyle: 'italic',
  },
  historyInterp: {
    fontSize: 9,
    color: '#666',
    marginTop: 6,
    paddingTop: 6,
    borderTopWidth: 1,
    borderTopColor: '#e8e8e8',
    lineHeight: 13,
  },

  accessDenied: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  },
  deniedTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#333',
  },
  deniedText: {
    fontSize: 13,
    color: '#666',
  },
  backBtn: {
    marginTop: 12,
    backgroundColor: theme.colors.green,
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
  },
  backBtnText: {
    color: '#fff',
    fontWeight: '600',
  },
});
