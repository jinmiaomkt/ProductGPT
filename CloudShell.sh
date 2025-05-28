# Inside CloudShell
mkdir -p ~/metrics
aws s3 cp s3://productgptbucket/DecisionOnly/metrics/ ~/metrics/ \
  --recursive --exclude "*" --include "DecisionOnly_performer*.json"

# Zip and download
cd ~
zip -r DecisionOnly.zip metrics
# Click the "Download" button that appears in CloudShell after the zip finishes

# Put it in a temporary folder in the same bucket
aws s3 cp DecisionOnly.zip s3://productgptbucket/tmp/DecisionOnly.zip



# Step 1: Create the destination folder
mkdir -p ~/metrics

# Step 2: Download JSON files from the correct S3 path with correct pattern
aws s3 cp s3://productgptbucket/DecisionOnly/metrics/ ~/metrics/ \
  --recursive --exclude "*" --include "DecisionOnly_ctx*.json"

# Step 3: Zip the folder
cd ~
zip -r DecisionOnly.zip metrics

# Step 4: Upload zipped archive to temporary S3 location
aws s3 cp DecisionOnly.zip s3://productgptbucket/tmp/DecisionOnly.zip
