# Outputs

This directory will contain generated outputs from the pipeline, such as:

- outputs/forecast.png — visualization of actual vs forecast
- outputs/model.joblib — serialized trained model
- outputs/metrics.txt — MAE and RMSE numbers

Run the pipeline script to populate this directory:

python src/forecasting_pipeline.py --data data/sample_sales.csv --output outputs
