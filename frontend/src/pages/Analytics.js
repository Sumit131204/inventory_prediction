import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  TextField,
  Button,
  CircularProgress,
  Alert,
  Chip,
} from '@mui/material';
import {
  Search as SearchIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const Analytics = () => {
  const [formData, setFormData] = useState({
    store: 1,
    item: 1,
    days: 90,
  });
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleAnalyze = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await axios.get(
        `${API_BASE_URL}/analytics/${formData.store}/${formData.item}`,
        {
          params: { days: formData.days },
        }
      );

      setAnalytics(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch analytics');
      console.error('Analytics error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getTrendColor = (trend) => {
    switch (trend) {
      case 'increasing':
        return 'success';
      case 'decreasing':
        return 'error';
      case 'stable':
        return 'info';
      default:
        return 'default';
    }
  };

  const getTrendLabel = (trend) => {
    switch (trend) {
      case 'increasing':
        return '📈 Increasing';
      case 'decreasing':
        return '📉 Decreasing';
      case 'stable':
        return '➡️ Stable';
      default:
        return '❓ Unknown';
    }
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" gutterBottom>
          Analytics
        </Typography>
        <Typography variant="body1" color="textSecondary">
          Historical sales analysis and trends
        </Typography>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Analysis Parameters
            </Typography>

            <Box sx={{ mt: 3, display: 'flex', flexDirection: 'column', gap: 2 }}>
              <TextField
                label="Store"
                name="store"
                type="number"
                value={formData.store}
                onChange={handleInputChange}
                inputProps={{ min: 1, max: 10 }}
                fullWidth
              />

              <TextField
                label="Item"
                name="item"
                type="number"
                value={formData.item}
                onChange={handleInputChange}
                inputProps={{ min: 1, max: 50 }}
                fullWidth
              />

              <TextField
                label="Days of History"
                name="days"
                type="number"
                value={formData.days}
                onChange={handleInputChange}
                inputProps={{ min: 30, max: 365 }}
                fullWidth
              />

              <Button
                variant="contained"
                size="large"
                startIcon={loading ? <CircularProgress size={20} /> : <SearchIcon />}
                onClick={handleAnalyze}
                disabled={loading}
                fullWidth
              >
                {loading ? 'Analyzing...' : 'Analyze'}
              </Button>
            </Box>

            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={8}>
          {analytics ? (
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Paper sx={{ p: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6">
                      Store {analytics.store} • Item {analytics.item}
                    </Typography>
                    <Chip
                      label={getTrendLabel(analytics.trend)}
                      color={getTrendColor(analytics.trend)}
                    />
                  </Box>
                  
                  <Grid container spacing={2} sx={{ mt: 1 }}>
                    <Grid item xs={6} sm={4}>
                      <Box>
                        <Typography variant="body2" color="textSecondary">
                          Mean Sales
                        </Typography>
                        <Typography variant="h6">
                          {analytics.statistics.mean_sales.toFixed(2)}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={4}>
                      <Box>
                        <Typography variant="body2" color="textSecondary">
                          Median Sales
                        </Typography>
                        <Typography variant="h6">
                          {analytics.statistics.median_sales.toFixed(2)}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={4}>
                      <Box>
                        <Typography variant="body2" color="textSecondary">
                          Std Dev
                        </Typography>
                        <Typography variant="h6">
                          {analytics.statistics.std_sales.toFixed(2)}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={4}>
                      <Box>
                        <Typography variant="body2" color="textSecondary">
                          Min Sales
                        </Typography>
                        <Typography variant="h6">
                          {analytics.statistics.min_sales.toFixed(2)}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={4}>
                      <Box>
                        <Typography variant="body2" color="textSecondary">
                          Max Sales
                        </Typography>
                        <Typography variant="h6">
                          {analytics.statistics.max_sales.toFixed(2)}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={4}>
                      <Box>
                        <Typography variant="body2" color="textSecondary">
                          Total Sales
                        </Typography>
                        <Typography variant="h6">
                          {analytics.statistics.total_sales.toFixed(0)}
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </Paper>
              </Grid>

              <Grid item xs={12}>
                <Paper sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Sales History
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={analytics.historical_data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="date"
                        tickFormatter={(value) => {
                          const date = new Date(value);
                          return `${date.getMonth() + 1}/${date.getDate()}`;
                        }}
                      />
                      <YAxis />
                      <Tooltip
                        labelFormatter={(value) => new Date(value).toLocaleDateString()}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="sales"
                        stroke="#1e3a8a"
                        strokeWidth={2}
                        dot={false}
                        name="Sales"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
            </Grid>
          ) : (
            <Paper sx={{ p: 4, textAlign: 'center' }}>
              <Typography variant="h6" color="textSecondary">
                Enter analysis parameters and click "Analyze" to see results
              </Typography>
            </Paper>
          )}
        </Grid>
      </Grid>
    </Container>
  );
};

export default Analytics;
