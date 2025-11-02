# Quick Deployment Steps

## 🚀 Backend → Render (5 minutes)

1. **Go to** [dashboard.render.com](https://dashboard.render.com)
2. **Click** "New +" → "Web Service"
3. **Connect** GitHub repository
4. **Configure:**
   - Name: `inventory-prediction-api`
   - Build: `pip install -r backend/requirements.txt`
   - Start: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
5. **Add Env Var:** `ENVIRONMENT` = `production`
6. **Deploy** → Copy backend URL (e.g., `https://xxx.onrender.com`)

## 🌐 Frontend → Vercel (3 minutes)

1. **Go to** [vercel.com](https://vercel.com)
2. **Click** "Add New..." → "Project"
3. **Import** from GitHub
4. **Configure:**
   - Root Directory: `frontend`
   - Framework: `Create React App`
5. **Add Env Var:** `REACT_APP_API_URL` = `https://your-backend-url.onrender.com`
6. **Deploy** → Done!

## 📤 Upload Models & Data

See `UPLOAD_MODELS_AND_DATA.md` for detailed instructions.

**Quick method:**
1. Render Dashboard → Your Service → "Shell"
2. Upload via `scp` or use Render Shell to transfer files

---

## ✅ Verify Deployment

**Backend:**
- Visit: `https://your-backend.onrender.com/docs`
- Test: `https://your-backend.onrender.com/health`

**Frontend:**
- Visit: `https://your-frontend.vercel.app`
- Should connect to backend automatically

---

**Full guide:** See `DEPLOYMENT_GUIDE.md` for detailed steps and troubleshooting.

