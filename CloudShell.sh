# 1. Start completely clean
rm -rf ~/metrics
mkdir -p ~/metrics

# Step 2: Download JSON files from the correct S3 path with correct pattern
# aws s3 cp s3://productgptbucket/DecisionOnly/metrics/ ~/metrics/ --recursive --exclude "*" --include "*/DecisionOnly_*.json"
aws s3 cp s3://productgptbucket/DecisionOnly/metrics/ ~/metrics/ --recursive --exclude "*" --include "DecisionOnly_*.json"

# Step 3: Zip the folder
cd ~
zip -r DecisionOnly.zip metrics

# Step 4: Upload zipped archive to temporary S3 location
aws s3 cp DecisionOnly.zip s3://productgptbucket/tmp/DecisionOnly.zip 




# Step 1: Create the destination folder
rm -rf ~/metrics
mkdir -p ~/metrics

# Step 2: Download JSON files from the correct S3 path with correct pattern
aws s3 cp s3://productgptbucket/FullProductGPT/performer/IndexBased/metrics/ ~/metrics/ --recursive --exclude "*" --include "FullProductGPT_indexbased*.json"
# aws s3 cp s3://productgptbucket/FullProductGPT/performer/Index/metrics/ ~/metrics/ --recursive --exclude "*" --include "*/FullProductGPT_indexbased*.json"

# Step 3: Zip the folder
cd ~
zip -r IndexBased.zip metrics

# Step 4: Upload zipped archive to temporary S3 location
aws s3 cp IndexBased.zip s3://productgptbucket/tmp/IndexBased.zip 






# Step 0 – Clean slate: remove any leftovers from previous runs
rm -rf ~/metrics
mkdir -p ~/metrics

# Step 1 – Preview the copy so you can confirm the matches
#     If it looks right, run the same command again without --dryrun:

aws s3 cp s3://productgptbucket/FullProductGPT/performer/FeatureBased/metrics/ \
         ~/metrics/ \
         --recursive \
         --exclude "*" \
         --include "*FullProductGPT_featurebased*.json"

# Step 2 – Zip what you just downloaded
cd ~
zip -r FeatureBased.zip metrics

# Step 3 – Upload the archive somewhere temporary
aws s3 cp FeatureBased.zip s3://productgptbucket/tmp/FeatureBased.zip




#!/usr/bin/env bash
set -euo pipefail

BUCKET="productgptbucket"
PREFIX="FullProductGPT/performer/FeatureBased/metrics"
PATTERN="FullProductGPT_featurebased*.json"   # <— corrected pattern

LOCAL_DIR="$HOME/metrics"
ZIP_NAME="FeatureBased.zip"


# 1. clean workspace
rm -rf "$LOCAL_DIR"mkdir -p "$LOCAL_DIR"

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

