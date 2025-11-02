# How to Upload Models and Data to Render

After deploying your backend to Render, you need to upload your ML models and processed data files.

## Option 1: Using Render Shell (SSH) - Recommended

### Step 1: Access Render Shell
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Select your web service (inventory-prediction-api)
3. Click on the **"Shell"** tab
4. You'll get a terminal prompt

### Step 2: Navigate to Project Directory
```bash
cd /opt/render/project/src
```

### Step 3: Create Required Directories
```bash
mkdir -p models
mkdir -p data/processed
```

### Step 4: Upload Files Using SCP (from your local machine)

Open a new terminal on your local machine and run:

```bash
# Upload models
scp -r models/*.pkl your-username@your-service.onrender.com:/opt/render/project/src/models/

# Upload processed data
scp data/processed/processed_data.csv your-username@your-service.onrender.com:/opt/render/project/src/data/processed/
```

**Note:** Replace `your-username` and `your-service.onrender.com` with your actual Render credentials. You can find SSH details in Render Dashboard → Your Service → "Info" tab.

## Option 2: Using Git (for smaller files)

If your models are small enough (< 100MB total), you can:

1. Temporarily remove models from `.gitignore`
2. Add and commit models:
   ```bash
   git add models/*.pkl
   git commit -m "Add model files for deployment"
   git push
   ```
3. Render will automatically redeploy
4. Re-add models to `.gitignore` after deployment

**⚠️ Warning:** This is not recommended for large files as it bloats your Git repository.

## Option 3: Using Cloud Storage (Best for Production)

For production deployments, store models in cloud storage:

1. **Upload to AWS S3 / Google Cloud Storage / Azure Blob Storage**
2. **Download on startup** - Modify `backend/main.py` to download models on startup:

```python
import boto3
import os

def download_models_from_s3():
    s3 = boto3.client('s3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    bucket_name = os.getenv('S3_BUCKET_NAME')
    
    # Download all .pkl files
    for key in s3.list_objects_v2(Bucket=bucket_name, Prefix='models/')['Contents']:
        filename = key['Key'].split('/')[-1]
        s3.download_file(bucket_name, key['Key'], f'models/{filename}')
```

Then add to `@app.on_event("startup")` before loading models.

## Option 4: Using Render Disks (For Persistent Storage)

1. **In Render Dashboard → Your Service → "Disks"**
2. **Add a new disk:**
   - Name: `models-disk`
   - Mount Path: `/opt/render/project/src/models`
   - Size: 1-5 GB (depending on model sizes)

3. **Upload files via Shell:**
   - Files uploaded to mounted disk path will persist across deployments

## Verification

After uploading, restart your service and check logs:

1. **Restart Service**: Render Dashboard → Your Service → "Manual Deploy" → "Clear build cache & deploy"
2. **Check Logs**: Should see messages like:
   ```
   ✓ Loaded lightgbm model
   ✓ Loaded data with X records
   ```

3. **Test API**: Visit `https://your-service.onrender.com/health`
   - Should show `models_loaded: 6` (or however many you uploaded)

## Troubleshooting

**Problem: Files not found**
- Check file paths match exactly (case-sensitive)
- Verify files are in correct directory structure
- Check Render logs for actual paths being accessed

**Problem: Disk space issues**
- Render free tier has limited disk space
- Consider using cloud storage (Option 3) for larger models
- Compress models if possible using `joblib.dump()` with compression

**Problem: Models take too long to load**
- Consider lazy loading (load on first request)
- Use lighter models for faster startup
- Add caching layer

