# Deployment Guide: Inventory Prediction System

This guide will walk you through deploying the backend to **Render** and the frontend to **Vercel**.

---

## 📋 Prerequisites

1. **GitHub Repository**: Your project should be pushed to GitHub (follow previous steps if not done)
2. **Render Account**: Sign up at [render.com](https://render.com) (free tier available)
3. **Vercel Account**: Sign up at [vercel.com](https://vercel.com) (free tier available)
4. **Required Files**: Ensure you have:
   - `backend/main.py` - FastAPI application
   - `backend/requirements.txt` - Python dependencies
   - `frontend/package.json` - React dependencies
   - `models/` folder with trained models (`.pkl` files)
   - `data/processed/` folder with processed data CSV

---

## 🚀 Part 1: Deploy Backend to Render

### Step 1: Prepare Backend for Deployment

1. **Verify your backend structure:**
   ```
   backend/
   ├── main.py
   └── requirements.txt
   ```

2. **Ensure all dependencies are in `backend/requirements.txt`:**
   Your `backend/requirements.txt` should include:
   - fastapi
   - uvicorn[standard]
   - All ML libraries (pandas, numpy, scikit-learn, xgboost, lightgbm, joblib)
   - python-multipart

### Step 2: Create Render Configuration (Optional but Recommended)

Create a file named `render.yaml` in the root of your project:

```yaml
services:
  - type: web
    name: inventory-prediction-api
    env: python
    plan: free
    buildCommand: pip install -r backend/requirements.txt
    startCommand: cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: ENVIRONMENT
        value: production
    disk:
      name: models
      mountPath: /models
      sizeGB: 1
```

### Step 3: Deploy on Render

#### Option A: Using Render Dashboard (Recommended for beginners)

1. **Go to Render Dashboard:**
   - Visit [dashboard.render.com](https://dashboard.render.com)
   - Sign up/Login with GitHub

2. **Create New Web Service:**
   - Click **"New +"** → **"Web Service"**
   - Connect your GitHub repository
   - Select the repository: `inventory_prediction`

3. **Configure Service:**
   - **Name**: `inventory-prediction-api` (or your preferred name)
   - **Region**: Choose closest to your users (e.g., `Oregon (US West)`)
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty (or `backend` if you want to deploy only backend folder)
   - **Runtime**: `Python 3`
   - **Build Command**: 
     ```bash
     pip install -r backend/requirements.txt
     ```
   - **Start Command**: 
     ```bash
     cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
   
   **⚠️ Important:** Render provides `$PORT` environment variable automatically.

4. **Add Environment Variables:**
   - Click **"Advanced"** → **"Add Environment Variable"**
   - Add:
     - `ENVIRONMENT` = `production`
     - `PYTHON_VERSION` = `3.11.0` (optional, ensures Python version)

5. **Add Disk for Models (Important!):**
   - In **"Advanced"** settings, scroll to **"Disk"**
   - Click **"Add Disk"**
   - Name: `models-disk`
   - Mount Path: `/opt/render/project/src/models`
   - Size: `1 GB` (adjust based on model file sizes)
   
   **Note:** You may need to update `MODEL_DIR` path in `main.py` to use `/opt/render/project/src/models` if models are stored differently.

6. **Deploy:**
   - Click **"Create Web Service"**
   - Render will start building and deploying
   - Wait 5-10 minutes for first deployment

7. **Upload Models and Data:**
   - Once deployed, you need to upload your models and data
   - **Option 1**: Use Render Shell (SSH)
     - Go to your service → **"Shell"** tab
     - Navigate to project directory
     - Use `git clone` or `scp` to upload files
   
   - **Option 2**: Use GitHub Actions or CI/CD to automatically sync files
   
   - **Option 3**: Include in git (if files are small enough)
     - Update `.gitignore` to allow models (not recommended for large files)

8. **Get Your Backend URL:**
   - After successful deployment, Render will provide a URL like:
     `https://inventory-prediction-api.onrender.com`
   - Save this URL - you'll need it for frontend configuration
   - Test the API: Visit `https://your-url.onrender.com/docs` to see Swagger UI

#### Option B: Using Render CLI

```bash
# Install Render CLI
npm install -g render-cli

# Login
render login

# Deploy using render.yaml
render deploy
```

### Step 4: Verify Backend Deployment

1. **Test Health Endpoint:**
   ```
   GET https://your-backend-url.onrender.com/health
   ```
   Should return: `{"status": "healthy", ...}`

2. **Test API Docs:**
   Visit: `https://your-backend-url.onrender.com/docs`

3. **Check Logs:**
   - Go to Render Dashboard → Your Service → **"Logs"** tab
   - Verify models and data are loading correctly

---

## 🌐 Part 2: Deploy Frontend to Vercel

### Step 1: Prepare Frontend

1. **Verify `frontend/package.json` exists** with all dependencies

2. **Update Vercel Configuration:**
   Your `frontend/vercel.json` is already configured, but verify it looks correct:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "build"
      }
    }
  ],
  "routes": [
    {
      "src": "/static/(.*)",
      "dest": "/static/$1"
    },
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ]
}
```

3. **Create `.env.production` file** (optional, you'll set in Vercel dashboard):
   ```
   REACT_APP_API_URL=https://your-backend-url.onrender.com
   ```

### Step 2: Deploy on Vercel

#### Option A: Using Vercel Dashboard

1. **Go to Vercel Dashboard:**
   - Visit [vercel.com](https://vercel.com)
   - Sign up/Login with GitHub

2. **Import Project:**
   - Click **"Add New..."** → **"Project"**
   - Import from GitHub
   - Select repository: `inventory_prediction`

3. **Configure Project:**
   - **Framework Preset**: `Create React App`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build` (auto-detected)
   - **Output Directory**: `build` (auto-detected)
   - **Install Command**: `npm install` (auto-detected)

4. **Add Environment Variables:**
   - Click **"Environment Variables"**
   - Add:
     - **Key**: `REACT_APP_API_URL`
     - **Value**: `https://your-backend-url.onrender.com` (from Render deployment)
     - **Environment**: `Production`, `Preview`, `Development` (select all)
   - Click **"Save"**

5. **Deploy:**
   - Click **"Deploy"**
   - Wait 2-5 minutes for build and deployment

6. **Get Your Frontend URL:**
   - After deployment, Vercel will provide a URL like:
     `https://inventory-prediction-frontend.vercel.app`
   - You can also add a custom domain later

#### Option B: Using Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Navigate to frontend directory
cd frontend

# Deploy
vercel --prod

# Set environment variable
vercel env add REACT_APP_API_URL production
# Enter: https://your-backend-url.onrender.com
```

### Step 3: Update Backend CORS (If Needed)

If your frontend is on a custom domain, update backend CORS:

1. Go to Render Dashboard → Your Service → **"Environment"**
2. Add environment variable:
   - `ALLOWED_ORIGINS` = `https://your-frontend.vercel.app,https://your-custom-domain.com`

Or update `backend/main.py` to include your specific Vercel URL.

### Step 4: Verify Frontend Deployment

1. **Visit your Vercel URL**
2. **Test the application:**
   - Dashboard should load
   - API calls should work
   - Check browser console for errors

---

## 🔧 Troubleshooting

### Backend Issues

**Problem: Models not loading**
- **Solution**: Ensure models are in the correct path. Check Render logs for file paths.
- Update `MODEL_DIR` in `main.py` if needed

**Problem: Data file not found**
- **Solution**: Ensure `data/processed/processed_data.csv` exists on Render
- Upload via Render Shell or include in deployment

**Problem: CORS errors**
- **Solution**: Update CORS settings in `backend/main.py` to include your Vercel URL

**Problem: Build fails**
- **Solution**: Check `backend/requirements.txt` has all dependencies
- Check Render logs for specific error messages

### Frontend Issues

**Problem: API calls fail**
- **Solution**: 
  - Verify `REACT_APP_API_URL` is set correctly in Vercel
  - Check backend is running (visit backend URL)
  - Check browser console for CORS errors

**Problem: Build fails**
- **Solution**: 
  - Check all dependencies in `package.json`
  - Verify Node.js version (Vercel uses Node 18 by default)
  - Check build logs in Vercel dashboard

**Problem: Routes don't work (404 errors)**
- **Solution**: Verify `vercel.json` has catch-all route to `index.html`

---

## 📝 Post-Deployment Checklist

- [ ] Backend deployed on Render
- [ ] Backend URL accessible (`/health` endpoint works)
- [ ] Models and data files uploaded to Render
- [ ] Frontend deployed on Vercel
- [ ] `REACT_APP_API_URL` environment variable set
- [ ] Frontend can communicate with backend
- [ ] All features tested (predictions, analytics, etc.)
- [ ] Custom domain configured (optional)

---

## 🔄 Updating Deployments

### Update Backend:
1. Push changes to GitHub
2. Render auto-deploys (if auto-deploy is enabled)
3. Or manually trigger deployment in Render dashboard

### Update Frontend:
1. Push changes to GitHub
2. Vercel auto-deploys
3. Or use `vercel --prod` from CLI

---

## 💡 Tips

1. **Monitor Logs**: Regularly check logs on both Render and Vercel
2. **Set up Alerts**: Configure email notifications for deployment failures
3. **Use Custom Domains**: Add professional domains for production
4. **Enable Auto-Deploy**: Both platforms support auto-deploy from GitHub
5. **Free Tier Limits**: 
   - Render free tier sleeps after 15 minutes of inactivity
   - Vercel free tier is always-on with generous limits

---

## 📚 Additional Resources

- [Render Documentation](https://render.com/docs)
- [Vercel Documentation](https://vercel.com/docs)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [React Deployment](https://create-react-app.dev/docs/deployment/)

---

**Need Help?** Check the logs, verify environment variables, and ensure all files are in the correct locations.

