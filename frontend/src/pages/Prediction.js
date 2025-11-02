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
  MenuItem,
  Card,
  CardContent,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  Send as SendIcon,
  TrendingUp,
  Inventory2,
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
  Area,
  AreaChart,
} from 'recharts';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const Prediction = () => {
  const [formData, setFormData] = useState({
    store: 1,
    item: 1,
    date: new Date().toISOString().split('T')[0],
    model_name: 'lightgbm',
  });
  const [models, setModels] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [forecast, setForecast] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/models`);
      setModels(response.data.filter((m) => m.loaded));
    } catch (err) {
      console.error('Failed to fetch models:', err);
    }
  };

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handlePredict = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await axios.post(`${API_BASE_URL}/predict`, formData);
      setPrediction(response.data);

      // Fetch 7-day forecast
      const forecastResponse = await axios.get(
        `${API_BASE_URL}/forecast/${formData.store}/${formData.item}`,
        {
          params: {
            days: 7,
            model_name: formData.model_name,
          },
        }
      );

      setForecast(
        forecastResponse.data.predictions.map((p) => ({
          date: p.date,
          predicted: p.predicted_sales,
          lower: p.confidence_lower,
          upper: p.confidence_upper,
        }))
      );
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" gutterBottom>
          Inventory Prediction
        </Typography>
        <Typography variant="body1" color="textSecondary">
          Predict sales and get inventory recommendations
        </Typography>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Prediction Parameters
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
                label="Date"
                name="date"
                type="date"
                value={formData.date}
                onChange={handleInputChange}
                fullWidth
                InputLabelProps={{
                  shrink: true,
                }}
              />

              <TextField
                label="Model"
                name="model_name"
                select
                value={formData.model_name}
                onChange={handleInputChange}
                fullWidth
              >
                {models.map((model) => (
                  <MenuItem key={model.name} value={model.name}>
                    {model.type} (R² = {model.r2?.toFixed(3)})
                  </MenuItem>
                ))}
              </TextField>

              <Button
                variant="contained"
                size="large"
                startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
                onClick={handlePredict}
                disabled={loading}
                fullWidth
              >
                {loading ? 'Predicting...' : 'Predict'}
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
          {prediction ? (
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <TrendingUp sx={{ mr: 1, color: '#0891b2' }} />
                      <Typography variant="h6">Predicted Sales</Typography>
                    </Box>
                    <Typography variant="h3" color="primary">
                      {prediction.predicted_sales.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Confidence: {prediction.confidence_lower.toFixed(2)} -{' '}
                      {prediction.confidence_upper.toFixed(2)}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Inventory2 sx={{ mr: 1, color: '#f97316' }} />
                      <Typography variant="h6">Recommended Inventory</Typography>
                    </Box>
                    <Typography variant="h3" color="secondary">
                      {prediction.recommended_inventory}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Units (includes 20% safety stock)
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Paper sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Prediction Details
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" paragraph>
                      <strong>Store:</strong> {prediction.store}
                    </Typography>
                    <Typography variant="body2" paragraph>
                      <strong>Item:</strong> {prediction.item}
                    </Typography>
                    <Typography variant="body2" paragraph>
                      <strong>Date:</strong> {prediction.date}
                    </Typography>
                    <Typography variant="body2" paragraph>
                      <strong>Model:</strong> {prediction.model_used}
                    </Typography>
                  </Box>
                </Paper>
              </Grid>

              {forecast.length > 0 && (
                <Grid item xs={12}>
                  <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      7-Day Forecast
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={forecast}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Area
                          type="monotone"
                          dataKey="upper"
                          stroke="#0891b2"
                          fill="#cffafe"
                          name="Upper Bound"
                        />
                        <Area
                          type="monotone"
                          dataKey="predicted"
                          stroke="#1e3a8a"
                          fill="#dbeafe"
                          name="Predicted"
                        />
                        <Area
                          type="monotone"
                          dataKey="lower"
                          stroke="#0891b2"
                          fill="#f0fdfa"
                          name="Lower Bound"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </Paper>
                </Grid>
              )}
            </Grid>
          ) : (
            <Paper sx={{ p: 4, textAlign: 'center' }}>
              <Typography variant="h6" color="textSecondary">
                Enter prediction parameters and click "Predict" to see results
              </Typography>
            </Paper>
          )}
        </Grid>
      </Grid>
    </Container>
  );
};

export default Prediction;
