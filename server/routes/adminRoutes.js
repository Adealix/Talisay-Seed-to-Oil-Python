import { Router } from 'express';
import { requireAuth, requireAdmin } from '../lib/auth.js';
import * as adminController from '../controllers/adminController.js';

const router = Router();

// All admin routes require authentication + admin role
router.use(requireAuth, requireAdmin);

router.get('/users', adminController.listUsers);
router.get('/predictions', adminController.listAllPredictions);
router.get('/history', adminController.listAllHistory);
router.get('/analytics/overview', adminController.getAnalyticsOverview);
router.get('/analytics/charts', adminController.getChartData);

export default router;
