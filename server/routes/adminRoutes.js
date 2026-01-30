import { Router } from 'express';
import { requireAuth, requireAdmin } from '../lib/auth.js';
import * as adminController from '../controllers/adminController.js';

const router = Router();

// All admin routes require authentication + admin role
router.use(requireAuth, requireAdmin);

router.get('/users', adminController.listUsers);
router.get('/predictions', adminController.listAllPredictions);

export default router;
