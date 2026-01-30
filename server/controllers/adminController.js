import { User } from '../models/User.js';
import { Prediction } from '../models/Prediction.js';

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
 */
export async function listAllPredictions(req, res) {
  try {
    const items = await Prediction.find({}).sort({ createdAt: -1 }).limit(100).lean();
    return res.json({ ok: true, items });
  } catch (e) {
    console.error('[adminController.listAllPredictions]', e);
    return res.status(500).json({ ok: false, error: 'server_error' });
  }
}
