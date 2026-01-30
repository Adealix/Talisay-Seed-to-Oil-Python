import { Router } from 'express';
import { requireAuth } from '../lib/auth.js';
import * as authController from '../controllers/authController.js';

const router = Router();

// Public
router.post('/register', authController.register);
router.post('/login', authController.login);

// Protected
router.get('/me', requireAuth, authController.me);

export default router;
