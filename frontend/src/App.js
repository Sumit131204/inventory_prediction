import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box } from '@mui/material';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Prediction from './pages/Prediction';
import Analytics from './pages/Analytics';
import Models from './pages/Models';

function App() {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <Navbar />
      <Box component="main" sx={{ flexGrow: 1, p: 3, mt: 8 }}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/prediction" element={<Prediction />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/models" element={<Models />} />
        </Routes>
      </Box>
    </Box>
  );
}

export default App;
