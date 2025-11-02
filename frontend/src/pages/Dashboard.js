import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  TrendingUp,
  Inventory,
  Assessment,
  CheckCircle,
} from '@mui/icons-material';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const StatCard = ({ title, value, icon, color }) => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography color="textSecondary" gutterBottom variant="body2">
            {title}
          </Typography>
          <Typography variant="h4" component="div">
            {value}
          </Typography>
        </Box>
        <Box
          sx={{
            backgroundColor: color,
            borderRadius: '50%',
            p: 1.5,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {icon}
        </Box>
      </Box>
    </CardContent>
  </Card>
);

const Dashboard = () => {
  const [health, setHealth] = useState(null);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [healthRes, modelsRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/health`),
        axios.get(`${API_BASE_URL}/models`),
      ]);

      setHealth(healthRes.data);
      setModels(modelsRes.data);
      setError(null);
    } catch (err) {
      setError('Failed to load dashboard data. Please ensure the backend is running.');
      console.error('Dashboard error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Container sx={{ display: 'flex', justifyContent: 'center', mt: 8 }}>
        <CircularProgress />
      </Container>
    );
  }

  if (error) {
    return (
      <Container sx={{ mt: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  const loadedModels = models.filter((m) => m.loaded).length;
  const bestModel = models.reduce((best, model) =>
    model.r2 && (!best.r2 || model.r2 > best.r2) ? model : best
  , {});

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" gutterBottom>
          Dashboard
        </Typography>
        <Typography variant="body1" color="textSecondary">
          ML-based inventory prediction system overview
        </Typography>
      </Box>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="System Status"
            value={health?.status === 'healthy' ? 'Healthy' : 'Error'}
            icon={<CheckCircle sx={{ color: 'white' }} />}
            color="#16a34a"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Models Loaded"
            value={`${loadedModels} / ${models.length}`}
            icon={<Assessment sx={{ color: 'white' }} />}
            color="#0891b2"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Best Model"
            value={bestModel.name || 'N/A'}
            icon={<TrendingUp sx={{ color: 'white' }} />}
            color="#f97316"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Best Accuracy"
            value={bestModel.r2 ? `${(bestModel.r2 * 100).toFixed(1)}%` : 'N/A'}
            icon={<Inventory sx={{ color: 'white' }} />}
            color="#1e3a8a"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              System Information
            </Typography>
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" paragraph>
                <strong>Status:</strong> {health?.status}
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Models Loaded:</strong> {health?.models_loaded}
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Data Loaded:</strong> {health?.data_loaded ? 'Yes' : 'No'}
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Last Updated:</strong>{' '}
                {new Date(health?.timestamp).toLocaleString()}
              </Typography>
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Quick Stats
            </Typography>
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" paragraph>
                <strong>Stores:</strong> 10 locations
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Items:</strong> 50 products
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Dataset:</strong> 913,000+ records
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Time Period:</strong> 5 years (2013-2017)
              </Typography>
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Model Performance
            </Typography>
            <Box sx={{ overflowX: 'auto', mt: 2 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f3f4f6' }}>
                    <th style={{ padding: '12px', textAlign: 'left' }}>Model</th>
                    <th style={{ padding: '12px', textAlign: 'left' }}>Type</th>
                    <th style={{ padding: '12px', textAlign: 'right' }}>RMSE</th>
                    <th style={{ padding: '12px', textAlign: 'right' }}>MAE</th>
                    <th style={{ padding: '12px', textAlign: 'right' }}>R²</th>
                    <th style={{ padding: '12px', textAlign: 'right' }}>MAPE</th>
                    <th style={{ padding: '12px', textAlign: 'center' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((model, index) => (
                    <tr
                      key={model.name}
                      style={{
                        backgroundColor: index % 2 === 0 ? 'white' : '#f9fafb',
                      }}
                    >
                      <td style={{ padding: '12px', fontWeight: 500 }}>{model.name}</td>
                      <td style={{ padding: '12px' }}>{model.type}</td>
                      <td style={{ padding: '12px', textAlign: 'right' }}>
                        {model.rmse?.toFixed(2) || 'N/A'}
                      </td>
                      <td style={{ padding: '12px', textAlign: 'right' }}>
                        {model.mae?.toFixed(2) || 'N/A'}
                      </td>
                      <td style={{ padding: '12px', textAlign: 'right' }}>
                        {model.r2?.toFixed(3) || 'N/A'}
                      </td>
                      <td style={{ padding: '12px', textAlign: 'right' }}>
                        {model.mape?.toFixed(2) || 'N/A'}%
                      </td>
                      <td style={{ padding: '12px', textAlign: 'center' }}>
                        <span
                          style={{
                            padding: '4px 8px',
                            borderRadius: '4px',
                            backgroundColor: model.loaded ? '#dcfce7' : '#fee2e2',
                            color: model.loaded ? '#166534' : '#991b1b',
                            fontSize: '0.875rem',
                          }}
                        >
                          {model.loaded ? 'Loaded' : 'Not Loaded'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;
