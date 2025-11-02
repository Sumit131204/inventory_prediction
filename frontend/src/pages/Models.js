import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Chip,
} from '@mui/material';
import {
  CheckCircle,
  Cancel,
  TrendingUp,
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const Models = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/models`);
      setModels(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to load models');
      console.error('Models error:', err);
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

  const chartData = models
    .filter((m) => m.r2)
    .map((m) => ({
      name: m.name,
      r2: m.r2,
      rmse: m.rmse,
      mae: m.mae,
      mape: m.mape,
    }));

  const getModelColor = (index) => {
    const colors = ['#1e3a8a', '#0891b2', '#16a34a', '#f97316', '#dc2626', '#a855f7'];
    return colors[index % colors.length];
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" gutterBottom>
          Model Comparison
        </Typography>
        <Typography variant="body1" color="textSecondary">
          Performance metrics and comparison across all ML models
        </Typography>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              R² Score Comparison (Higher is Better)
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="r2" name="R² Score" radius={[8, 8, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getModelColor(index)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              RMSE Comparison (Lower is Better)
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="rmse" name="RMSE" radius={[8, 8, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getModelColor(index)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {models.map((model, index) => (
          <Grid item xs={12} md={6} key={model.name}>
            <Paper sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">{model.type}</Typography>
                <Chip
                  icon={model.loaded ? <CheckCircle /> : <Cancel />}
                  label={model.loaded ? 'Loaded' : 'Not Loaded'}
                  color={model.loaded ? 'success' : 'error'}
                />
              </Box>

              <Box sx={{ mt: 2 }}>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">
                      RMSE
                    </Typography>
                    <Typography variant="h6">
                      {model.rmse?.toFixed(2) || 'N/A'}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">
                      MAE
                    </Typography>
                    <Typography variant="h6">
                      {model.mae?.toFixed(2) || 'N/A'}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">
                      R² Score
                    </Typography>
                    <Typography variant="h6">
                      {model.r2?.toFixed(3) || 'N/A'}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">
                      MAPE
                    </Typography>
                    <Typography variant="h6">
                      {model.mape?.toFixed(2) || 'N/A'}%
                    </Typography>
                  </Grid>
                </Grid>

                {model.r2 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                      Performance Rating
                    </Typography>
                    <Box
                      sx={{
                        width: '100%',
                        height: 8,
                        backgroundColor: '#e5e7eb',
                        borderRadius: 1,
                        overflow: 'hidden',
                      }}
                    >
                      <Box
                        sx={{
                          width: `${model.r2 * 100}%`,
                          height: '100%',
                          backgroundColor: getModelColor(index),
                        }}
                      />
                    </Box>
                    <Typography variant="caption" color="textSecondary">
                      {(model.r2 * 100).toFixed(1)}% variance explained
                    </Typography>
                  </Box>
                )}
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
};

export default Models;
