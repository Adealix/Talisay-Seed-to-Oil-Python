import bcrypt from 'bcryptjs';
import { User } from '../models/User.js';
import { signToken } from '../lib/auth.js';

/**
 * POST /api/auth/register
 * Create a new user account.
 */
export async function register(req, res) {
  try {
    const { email, password, role } = req.body ?? {};

    if (!email || !password) {
      return res.status(400).json({ ok: false, error: 'email_and_password_required' });
    }
    if (String(password).length < 6) {
      return res.status(400).json({ ok: false, error: 'password_too_short' });
    }

    const nextRole = role === 'admin' ? 'admin' : 'user';

    const passwordHash = await bcrypt.hash(String(password), 10);
    const user = await User.create({
      email: String(email).toLowerCase().trim(),
      passwordHash,
      role: nextRole,
    });

    const token = signToken(user);

    return res.json({
      ok: true,
      token,
      user: { id: String(user._id), email: user.email, role: user.role },
    });
  } catch (e) {
    // Duplicate key (email already exists)
    if (e?.code === 11000) {
      return res.status(409).json({ ok: false, error: 'email_already_exists' });
    }
    console.error('[authController.register]', e);
    return res.status(500).json({ ok: false, error: 'server_error' });
  }
}

/**
 * POST /api/auth/login
 * Authenticate user and return JWT.
 */
export async function login(req, res) {
  try {
    const { email, password } = req.body ?? {};

    if (!email || !password) {
      return res.status(400).json({ ok: false, error: 'email_and_password_required' });
    }

    const user = await User.findOne({ email: String(email).toLowerCase().trim() });
    if (!user) {
      return res.status(401).json({ ok: false, error: 'invalid_credentials' });
    }

    const valid = await bcrypt.compare(String(password), user.passwordHash);
    if (!valid) {
      return res.status(401).json({ ok: false, error: 'invalid_credentials' });
    }

    const token = signToken(user);
    return res.json({
      ok: true,
      token,
      user: { id: String(user._id), email: user.email, role: user.role },
    });
  } catch (e) {
    console.error('[authController.login]', e);
    return res.status(500).json({ ok: false, error: 'server_error' });
  }
}

/**
 * GET /api/auth/me
 * Return the currently authenticated user.
 */
export async function me(req, res) {
  try {
    const user = await User.findById(req.auth.userId).lean();
    if (!user) {
      return res.status(404).json({ ok: false, error: 'not_found' });
    }
    return res.json({
      ok: true,
      user: { id: String(user._id), email: user.email, role: user.role },
    });
  } catch (e) {
    console.error('[authController.me]', e);
    return res.status(500).json({ ok: false, error: 'server_error' });
  }
}
