import { User } from '../models/User.js';
import { History } from '../models/History.js';
// Note: Prediction model is no longer used - all predictions are stored in History

/**
 * GET /api/admin/users
 * List all users (admin only).
 */
export async function listUsers(req, res) {
  try {
    const users = await User.find({}).sort({ createdAt: -1 }).limit(100).lean();
    return res.json({
      ok: true,
      users: users.map((u) => ({
        id: String(u._id),
        email: u.email,
        role: u.role,
        createdAt: u.createdAt,
      })),
    });
  } catch (e) {
    console.error('[adminController.listUsers]', e);
    return res.status(500).json({ ok: false, error: 'server_error' });
  }
}

/**
 * GET /api/admin/predictions
 * List all predictions (admin only).
 * NOTE: Predictions now come from History collection (where scans are saved)
 */
export async function listAllPredictions(req, res) {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 100, 500);
    
    const items = await History.find({})
      .populate('userId', 'email')
      .sort({ createdAt: -1 })
      .limit(limit)
      .lean();

    // Map to prediction-like format for compatibility
    return res.json({ 
      ok: true, 
      items: items.map(item => ({
        _id: String(item._id),
        userId: String(item.userId?._id || item.userId),
        userEmail: item.userId?.email || 'Unknown',
        category: item.category,
        confidence: item.confidence,
        ratio: item.ratio,
        oilYieldPercent: item.oilYieldPercent,
        yieldCategory: item.yieldCategory,
        createdAt: item.createdAt,
      }))
    });
  } catch (e) {
    console.error('[adminController.listAllPredictions]', e);
    return res.status(500).json({ ok: false, error: 'server_error' });
  }
}

/**
 * GET /api/admin/history
 * List all history items from all users (admin only).
 */
export async function listAllHistory(req, res) {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 100, 500);
    
    const items = await History.find({})
      .populate('userId', 'email')
      .sort({ createdAt: -1 })
      .limit(limit)
      .lean();

    return res.json({ 
      ok: true, 
      items: items.map(item => ({
        ...item,
        id: String(item._id),
        _id: undefined,
        userEmail: item.userId?.email || 'Unknown',
        userId: String(item.userId?._id || item.userId),
      }))
    });
  } catch (e) {
    console.error('[adminController.listAllHistory]', e);
    return res.status(500).json({ ok: false, error: 'server_error' });
  }
}

/**
 * GET /api/admin/analytics/overview
 * Get comprehensive analytics overview (admin only).
 */
export async function getAnalyticsOverview(req, res) {
  try {
    // Get counts - totalPredictions now uses History since that's where scans are saved
    const [totalUsers, totalHistory] = await Promise.all([
      User.countDocuments({}),
      History.countDocuments({}),
    ]);
    
    // totalPredictions is the same as totalHistory (all predictions are stored as history)
    const totalPredictions = totalHistory;

    // Get category distribution
    const categoryDistribution = await History.aggregate([
      { $group: { _id: '$category', count: { $sum: 1 } } },
      { $sort: { count: -1 } }
    ]);

    // Get yield category distribution
    const yieldDistribution = await History.aggregate([
      { $match: { yieldCategory: { $ne: null } } },
      { $group: { _id: '$yieldCategory', count: { $sum: 1 } } },
      { $sort: { count: -1 } }
    ]);

    // Get average oil yield by category
    const avgYieldByCategory = await History.aggregate([
      { $match: { oilYieldPercent: { $ne: null } } },
      { 
        $group: { 
          _id: '$category', 
          avgYield: { $avg: '$oilYieldPercent' },
          minYield: { $min: '$oilYieldPercent' },
          maxYield: { $max: '$oilYieldPercent' },
          count: { $sum: 1 }
        } 
      },
      { $sort: { _id: 1 } }
    ]);

    // Get daily scan activity (last 30 days)
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
    
    const dailyActivity = await History.aggregate([
      { $match: { createdAt: { $gte: thirtyDaysAgo } } },
      {
        $group: {
          _id: { $dateToString: { format: '%Y-%m-%d', date: '$createdAt' } },
          count: { $sum: 1 }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    // Get user activity (top 10 most active users)
    const userActivity = await History.aggregate([
      { 
        $group: { 
          _id: '$userId', 
          scanCount: { $sum: 1 },
          lastScan: { $max: '$createdAt' }
        } 
      },
      { $sort: { scanCount: -1 } },
      { $limit: 10 },
      {
        $lookup: {
          from: 'users',
          localField: '_id',
          foreignField: '_id',
          as: 'user'
        }
      },
      { $unwind: { path: '$user', preserveNullAndEmptyArrays: true } },
      {
        $project: {
          userId: '$_id',
          email: '$user.email',
          scanCount: 1,
          lastScan: 1
        }
      }
    ]);

    // Get reference coin detection rate
    const coinDetectionStats = await History.aggregate([
      {
        $group: {
          _id: null,
          totalScans: { $sum: 1 },
          withCoin: { $sum: { $cond: ['$referenceDetected', 1, 0] } }
        }
      }
    ]);

    // Get average confidence scores
    const confidenceStats = await History.aggregate([
      { $match: { confidence: { $ne: null } } },
      {
        $group: {
          _id: null,
          avgConfidence: { $avg: '$confidence' },
          minConfidence: { $min: '$confidence' },
          maxConfidence: { $max: '$confidence' }
        }
      }
    ]);

    // Get oil confidence stats (new field)
    const oilConfidenceStats = await History.aggregate([
      { $match: { oilConfidence: { $ne: null } } },
      {
        $group: {
          _id: null,
          avgOilConfidence: { $avg: '$oilConfidence' },
        }
      }
    ]);

    // Get spot detection statistics
    const spotStats = await History.aggregate([
      {
        $group: {
          _id: null,
          totalScans: { $sum: 1 },
          withSpots: { $sum: { $cond: ['$hasSpots', 1, 0] } },
          avgSpotCoverage: { $avg: { $cond: ['$hasSpots', '$spotCoverage', null] } }
        }
      }
    ]);

    // Get average yield overall
    const avgYieldOverall = await History.aggregate([
      { $match: { oilYieldPercent: { $ne: null } } },
      {
        $group: {
          _id: null,
          avgYield: { $avg: '$oilYieldPercent' }
        }
      }
    ]);

    // Get dimensions statistics
    const dimensionStats = await History.aggregate([
      { $match: { 'dimensions.lengthCm': { $ne: null } } },
      {
        $group: {
          _id: '$category',
          avgLength: { $avg: '$dimensions.lengthCm' },
          avgWidth: { $avg: '$dimensions.widthCm' },
          avgWeight: { $avg: '$dimensions.wholeFruitWeightG' },
          count: { $sum: 1 }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    // Get weekly trend (last 12 weeks)
    const twelveWeeksAgo = new Date();
    twelveWeeksAgo.setDate(twelveWeeksAgo.getDate() - 84);
    
    const weeklyTrend = await History.aggregate([
      { $match: { createdAt: { $gte: twelveWeeksAgo } } },
      {
        $group: {
          _id: { $dateToString: { format: '%Y-W%V', date: '$createdAt' } },
          count: { $sum: 1 },
          avgYield: { $avg: '$oilYieldPercent' }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    // Get new users this month
    const startOfMonth = new Date();
    startOfMonth.setDate(1);
    startOfMonth.setHours(0, 0, 0, 0);
    
    const newUsersThisMonth = await User.countDocuments({
      createdAt: { $gte: startOfMonth }
    });

    return res.json({
      ok: true,
      analytics: {
        overview: {
          totalUsers,
          totalHistory,
          totalPredictions,
          newUsersThisMonth,
        },
        categoryDistribution: categoryDistribution.reduce((acc, item) => {
          acc[item._id] = item.count;
          return acc;
        }, {}),
        yieldDistribution: yieldDistribution.reduce((acc, item) => {
          acc[item._id] = item.count;
          return acc;
        }, {}),
        avgYieldByCategory: avgYieldByCategory.reduce((acc, item) => {
          acc[item._id] = {
            avg: Math.round(item.avgYield * 100) / 100,
            min: Math.round(item.minYield * 100) / 100,
            max: Math.round(item.maxYield * 100) / 100,
            count: item.count,
          };
          return acc;
        }, {}),
        dailyActivity,
        weeklyTrend,
        userActivity: userActivity.map(u => ({
          userId: String(u.userId),
          email: u.email || 'Unknown',
          scanCount: u.scanCount,
          lastScan: u.lastScan,
        })),
        coinDetection: coinDetectionStats[0] || { totalScans: 0, withCoin: 0 },
        confidenceStats: confidenceStats[0] || { avgConfidence: 0, minConfidence: 0, maxConfidence: 0 },
        oilConfidenceStats: oilConfidenceStats[0] || { avgOilConfidence: 0 },
        spotStats: spotStats[0] || { totalScans: 0, withSpots: 0, avgSpotCoverage: 0 },
        avgYieldOverall: avgYieldOverall[0]?.avgYield || 0,
        dimensionStats: dimensionStats.reduce((acc, item) => {
          acc[item._id] = {
            avgLength: Math.round((item.avgLength || 0) * 100) / 100,
            avgWidth: Math.round((item.avgWidth || 0) * 100) / 100,
            avgWeight: Math.round((item.avgWeight || 0) * 100) / 100,
            count: item.count,
          };
          return acc;
        }, {}),
      }
    });
  } catch (e) {
    console.error('[adminController.getAnalyticsOverview]', e);
    return res.status(500).json({ ok: false, error: 'server_error' });
  }
}

/**
 * GET /api/admin/analytics/charts
 * Get chart-specific data for visualization (admin only).
 */
export async function getChartData(req, res) {
  try {
    const { chartType } = req.query;

    switch (chartType) {
      case 'oilYieldTrend': {
        // Oil yield over time (by date)
        const data = await History.aggregate([
          { $match: { oilYieldPercent: { $ne: null } } },
          {
            $group: {
              _id: { $dateToString: { format: '%Y-%m-%d', date: '$createdAt' } },
              avgYield: { $avg: '$oilYieldPercent' },
              count: { $sum: 1 }
            }
          },
          { $sort: { _id: 1 } },
          { $limit: 30 }
        ]);
        return res.json({ ok: true, data });
      }

      case 'categoryTimeline': {
        // Category distribution over time
        const data = await History.aggregate([
          {
            $group: {
              _id: {
                date: { $dateToString: { format: '%Y-%m-%d', date: '$createdAt' } },
                category: '$category'
              },
              count: { $sum: 1 }
            }
          },
          { $sort: { '_id.date': 1 } },
          { $limit: 100 }
        ]);
        return res.json({ ok: true, data });
      }

      case 'dimensionCorrelation': {
        // Dimension vs yield correlation
        const data = await History.find({
          oilYieldPercent: { $ne: null },
          'dimensions.lengthCm': { $ne: null }
        })
          .select('dimensions.lengthCm dimensions.widthCm dimensions.wholeFruitWeightG oilYieldPercent category')
          .limit(200)
          .lean();
        return res.json({ ok: true, data });
      }

      default: {
        return res.status(400).json({ ok: false, error: 'invalid_chart_type' });
      }
    }
  } catch (e) {
    console.error('[adminController.getChartData]', e);
    return res.status(500).json({ ok: false, error: 'server_error' });
  }
}
