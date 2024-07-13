# Data version control (DVC)

## How to setup dvc - GDrive

```bash
pip install dvc
pip install
```

## How to setup dvc - Cloud
### Buckets in Google Cloud
We have two buckets: `mlops-lmu-data-bucket` for our data and `mlops-fakenews-models` for our best model.
Important commands:
```bash
gsutil mb gs://<bucket-name>        # create a bucket
gsutil ls                           # list all buckets
gsutil cp <file> gs://<bucket-name> # upload a file to the cloud
```

### DVC in Google Cloud
There are many reasons that you want to store your data in the cloud including:
- Easily being able to share
- Easily expand as you need more
- Data is stored in multiple locations, making sure that it is not lost in case of an emergency

**Limitation of dvc with Google Drive** is that we need to authenticate each time we try to pull or pull. GCP offers to use an API for that instead.

Setup:
```bash
pip install dvc-gs      # already in requirements.txt
dvc remote add -d remote_storage gs://mlops-lmu-data-bucket/ # changes storage from Drive to GCS
dvc remote modify remote_storage version_aware true          # support object versioning

# These commands changed .dvc/config, so commit that file, then
dvc push --no-run-cache # push data to cloud without cache file
dvc pull --no-run-cache # pull data without having to give credentials
```
This will change the default way that dvc handles data. Instead of just storing the latest version of the data as content-addressable storage it will now store the data as it looks in our local repository, which allows us to not only use dvc to download our data.
