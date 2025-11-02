import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Container,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Predictions as PredictionsIcon,
  Analytics as AnalyticsIcon,
  Psychology as ModelsIcon,
} from '@mui/icons-material';

const Navbar = () => {
  return (
    <AppBar position="fixed">
      <Container maxWidth="xl">
        <Toolbar disableGutters>
          <Typography
            variant="h6"
            noWrap
            component={RouterLink}
            to="/"
            sx={{
              mr: 4,
              fontWeight: 700,
              color: 'inherit',
              textDecoration: 'none',
            }}
          >
            📦 Inventory Prediction
          </Typography>

          <Box sx={{ flexGrow: 1, display: 'flex', gap: 2 }}>
            <Button
              component={RouterLink}
              to="/"
              sx={{ color: 'white' }}
              startIcon={<DashboardIcon />}
            >
              Dashboard
            </Button>
            <Button
              component={RouterLink}
              to="/prediction"
              sx={{ color: 'white' }}
              startIcon={<PredictionsIcon />}
            >
              Predict
            </Button>
            <Button
              component={RouterLink}
              to="/analytics"
              sx={{ color: 'white' }}
              startIcon={<AnalyticsIcon />}
            >
              Analytics
            </Button>
            <Button
              component={RouterLink}
              to="/models"
              sx={{ color: 'white' }}
              startIcon={<ModelsIcon />}
            >
              Models
            </Button>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default Navbar;
