#!/usr/bin/env bash
# --------------------------------------------------------------
# Pull ONLY the 166 "DecisionOnly_performer_nb_features*.json",
# zip them, upload the archive, and leave nothing extra behind.
# --------------------------------------------------------------
set -euo pipefail

BUCKET="productgptbucket"
PREFIX="DecisionOnly/metrics"          # S3 pseudo-folder
LOCAL_DIR="$HOME/metrics"              # scratch space
ZIP_NAME="DecisionOnly.zip"

# 1. Start completely clean
rm -rf "$LOCAL_DIR"
mkdir -p "$LOCAL_DIR"

# 2. Copy just the desired JSON files
#    • --recursive because some files live one level deep (date folders).
#    • --exclude "*" blocks everything first.
#    • --include "*/DecisionOnly_performer_nb_features*.json"
#      grabs files exactly one folder beneath the prefix.
#    • --include "DecisionOnly_performer_nb_features*.json"
#      grabs any that sit directly inside the prefix.
#    • --exclude "DecisionOnly_ctx*" prevents the ctx32/dmodel256 variants.
aws s3 cp "s3://$BUCKET/$PREFIX/" "$LOCAL_DIR/" \
  --recursive \
  --exclude "*" \
  --include "DecisionOnly_performer_nb_features*.json" \
  --include "*/DecisionOnly_performer_nb_features*.json" \
  --exclude "DecisionOnly_ctx*"

# 4. Zip the folder quietly (no extra top-level directory)
(cd "$LOCAL_DIR" && zip -q -r "../$ZIP_NAME" .)

# 5. Upload the archive back to S3
aws s3 cp "$HOME/$ZIP_NAME" "s3://$BUCKET/tmp/$ZIP_NAME"

echo "Finished. Archive stored at:  s3://$BUCKET/tmp/$ZIP_NAME"




# Step 1: Create the destination folder
mkdir -p ~/metrics

# Step 2: Download JSON files from the correct S3 path with correct pattern
aws s3 cp s3://productgptbucket/FullProductGPT/performer/Index/metrics/ ~/metrics/ \
  --recursive --exclude "*" --include "FullProductGPT_performer_nb_features*.json"

aws s3 cp s3://productgptbucket/FullProductGPT/performer/Index/metrics/ ~/metrics/ --recursive --exclude "*" --include "*/FullProductGPT_performer_nb_features*.json"

# Step 3: Zip the folder
cd ~
zip -r IndexBased.zip metrics

# Step 4: Upload zipped archive to temporary S3 location
aws s3 cp IndexBased.zip s3://productgptbucket/tmp/IndexBased.zip --dryrun






# Step 1: Create the destination folder
mkdir -p ~/metrics

# Step 2: Download JSON files from the correct S3 path with correct pattern
aws s3 cp s3://productgptbucket/FullProductGPT/performer/FeatureBased/metrics/ ~/metrics/  --recursive --exclude "*" --include "FullProductGPT_performer_nb*.json"

aws s3 cp s3://productgptbucket/FullProductGPT/performer/FeatureBased/metrics/ ~/metrics/ --recursive --exclude "*" --include "*/FullProductGPT_performer_nb*.json"

# Step 3: Zip the folder
cd ~
zip -r FeatureBased.zip metrics

# Step 4: Upload zipped archive to temporary S3 location
aws s3 cp FeatureBased.zip s3://productgptbucket/tmp/FeatureBased.zip 


#!/usr/bin/env bash
set -euo pipefail

BUCKET="productgptbucket"
PREFIX="FullProductGPT/performer/FeatureBased/metrics"
PATTERN="FullProductGPT_performer_nbfeat*.json"   # <— corrected pattern

LOCAL_DIR="$HOME/metrics"
ZIP_NAME="FeatureBased.zip"

# 1. clean workspace
rm -rf "$LOCAL_DIR"
mkdir -p "$LOCAL_DIR"

# 2. copy ONLY the desired JSON files
aws s3 cp "s3://$BUCKET/$PREFIX/" "$LOCAL_DIR/" \
  --recursive \
  --exclude "*" \
  --include "$PATTERN" \
  --include "*/$PATTERN"

# 3. sanity-check
fetched=$(find "$LOCAL_DIR" -type f -name "$PATTERN" | wc -l)
echo "Fetched $fetched files."

# 4. zip (-j = flat archive)
zip -q -r "$HOME/$ZIP_NAME" -j "$LOCAL_DIR"

# 5. upload
aws s3 cp "$HOME/$ZIP_NAME" "s3://$BUCKET/tmp/$ZIP_NAME"

