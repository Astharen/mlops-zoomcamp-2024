export S3_ENDPOINT_URL="http://localhost:4566"
export PREDICTIONS_STREAM_NAME="ride_predictions"
export INPUT_FILE_PATTERN="s3://nyc-duration/year={year:04d}/month={month:02d}/input.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet"

python tests/integration_test.py 2023 1
