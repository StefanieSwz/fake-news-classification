# Monitoring

We implemented monitoring for data drift of our textual data (also in form of their embedding values) and label distribution coming from a reference database and a monitoring database originating from our inference app. The report can be accessed via a FastAPI application that creates reports and also downloads them.

For the inference app, we collect a broad set of metrics via the prometheus package. Most of the metrics are coming from a default setting, and one metric (number of made predictions) is manually implemented via incremention steps. If wished, the metrics can be accessed through a metrics/ endpoint in our inference FastAPI app.

On Google Cloud Platform, we tested the usability of the Monitoring Dashboard and created a widget for the CPU utilization and GPU RAM usage. We also created an alert system for a heavy request load to our Google Cloud Buckets, which was triggered once for now.

We added SLO to the Cloud Run "backend" to check the latency of the response, requiring that 80% of reponses must be completed in max 5 seconds. We are also able to stress test our API using locust, spawning multiple users and requests at the same time.
