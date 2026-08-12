# Cost Analysis Documentation

LanceDB reduces infrastructure storage cost because LanceDB uses disk-backed zero-copy storage on local EBS or S3 without requiring always-on RAM pods.
In contrast, managed cloud vector databases charge $45 to $70 per pod/month for keeping index vectors resident in RAM 24/7.
