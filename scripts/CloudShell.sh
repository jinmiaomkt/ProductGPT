#!/usr/bin/env bash
# ==========================================================
#  Build DecisionOnly.zip and upload it to S3
#  • Easy-to-edit variables at the top
#  • Linear “step-by-step” flow, no functions
# ==========================================================

# ---------- tweak these paths & names if needed ----------
BUCKET="productgptbucket"           # S3 bucket name
SRC_PREFIX="DecisionOnly/metrics"   # where the JSON lives in the bucket
DST_DIR="$HOME/metrics"             # local working folder
ZIP_FILE="$HOME/DecisionOnly.zip"   # archive name
FILE_MASK="DecisionOnly_*.json"     # pattern to keep
TMP_PREFIX="tmp"                    # S3 folder for the zip
# ---------------------------------------------------------

echo
echo "Step 1  –  clean workspace"
rm -rf "$DST_DIR"
mkdir -p "$DST_DIR"

echo
echo "Step 2  –  download only $FILE_MASK"
aws s3 cp "s3://${BUCKET}/${SRC_PREFIX}/" "$DST_DIR/" \
  --recursive \
  --exclude "*" \
  --include "$FILE_MASK"

echo
echo "Step 3  –  zip the matching files"
cd "$DST_DIR"
zip -j "$ZIP_FILE" $FILE_MASK        # -j strips path so the zip is flat
cd - >/dev/null

echo
echo "Step 4  –  upload archive to s3://${BUCKET}/${TMP_PREFIX}/"
aws s3 cp "$ZIP_FILE" \
  "s3://${BUCKET}/${TMP_PREFIX}/$(basename "$ZIP_FILE")"

echo
echo "Done!  Archive contains:"
zipinfo -1 "$ZIP_FILE"



# # Step 1: Create the destination folder
# rm -rf ~/metrics
# mkdir -p ~/metrics

# # Step 2: Download JSON files from the correct S3 path with correct pattern
# aws s3 cp s3://productgptbucket/FullProductGPT/performer/IndexBased/metrics/ ~/metrics/ --recursive --exclude "*" --include "FullProductGPT_indexbased*.json"
# # aws s3 cp s3://productgptbucket/FullProductGPT/performer/Index/metrics/ ~/metrics/ --recursive --exclude "*" --include "*/FullProductGPT_indexbased*.json"

# # Step 3: Zip the folder
# cd ~
# zip -r IndexBased.zip metrics

# # Step 4: Upload zipped archive to temporary S3 location
# aws s3 cp IndexBased.zip s3://productgptbucket/tmp/IndexBased.zip 

# ---------- tweak these paths & names if needed ----------
BUCKET="productgptbucket"           # S3 bucket name
SRC_PREFIX="FullProductGPT/performer/IndexBased/metrics/"   # where the JSON lives in the bucket
DST_DIR="$HOME/metrics"             # local working folder
ZIP_FILE="$HOME/IndexBased.zip"   # archive name
FILE_MASK="FullProductGPT_indexbased*.json"     # pattern to keep
TMP_PREFIX="tmp"                    # S3 folder for the zip
# ---------------------------------------------------------


echo
echo "Step 1  –  clean workspace"
rm -rf "$DST_DIR"
mkdir -p "$DST_DIR"

echo
echo "Step 2  –  download only $FILE_MASK"
aws s3 cp "s3://${BUCKET}/${SRC_PREFIX}/" "$DST_DIR/" \
  --recursive \
  --exclude "*" \
  --include "$FILE_MASK"

echo
echo "Step 3  –  zip the matching files"
cd "$DST_DIR"
zip -j "$ZIP_FILE" $FILE_MASK        # -j strips path so the zip is flat
cd - >/dev/null

echo
echo "Step 4  –  upload archive to s3://${BUCKET}/${TMP_PREFIX}/"
aws s3 cp "$ZIP_FILE" \
  "s3://${BUCKET}/${TMP_PREFIX}/$(basename "$ZIP_FILE")"

echo
echo "Done!  Archive contains:"
zipinfo -1 "$ZIP_FILE"


# ---------- tweak these paths & names if needed ----------
BUCKET="productgptbucket"           # S3 bucket name
SRC_PREFIX="FullProductGPT/performer/FeatureBased/metrics/"   # where the JSON lives in the bucket
DST_DIR="$HOME/metrics"             # local working folder
ZIP_FILE="$HOME/FeatureBased.zip"   # archive name
FILE_MASK="FullProductGPT_featurebased*.json"     # pattern to keep
TMP_PREFIX="tmp"                    # S3 folder for the zip
# ---------------------------------------------------------


echo
echo "Step 1  –  clean workspace"
rm -rf "$DST_DIR"
mkdir -p "$DST_DIR"

echo
echo "Step 2  –  download only $FILE_MASK"
aws s3 cp "s3://${BUCKET}/${SRC_PREFIX}/" "$DST_DIR/" \
  --recursive \
  --exclude "*" \
  --include "$FILE_MASK"

echo
echo "Step 3  –  zip the matching files"
cd "$DST_DIR"
zip -j "$ZIP_FILE" $FILE_MASK        # -j strips path so the zip is flat
cd - >/dev/null

echo
echo "Step 4  –  upload archive to s3://${BUCKET}/${TMP_PREFIX}/"
aws s3 cp "$ZIP_FILE" \
  "s3://${BUCKET}/${TMP_PREFIX}/$(basename "$ZIP_FILE")"

echo
echo "Done!  Archive contains:"
zipinfo -1 "$ZIP_FILE"


