WITH price_data AS (
  SELECT
    Symbol,
    index,
    Close,
    Close - LAG(Close) OVER (ORDER BY index) AS price_diff
  FROM `ambient-decoder-391319.stock_output.predicted_prices`
),
rsi_data AS (
  SELECT
  Symbol,
    index,
    Close,
    CASE WHEN price_diff > 0 THEN price_diff ELSE 0 END AS gain,
    CASE WHEN price_diff < 0 THEN ABS(price_diff) ELSE 0 END AS loss
  FROM price_data
),
histogram_data AS (
  SELECT
    Symbol,
    index,
    Close,
    gain,
    loss,
    NTILE(5) OVER (ORDER BY Close) AS bucket_number
  FROM rsi_data
)
SELECT
  Symbol,
  index,
  Close,
  CASE
    WHEN avg_gain IS NULL OR avg_loss IS NULL THEN NULL
    ELSE 100 - (100 / (1 + (NULLIF(avg_gain, 0) / NULLIF(avg_loss, 0))))
  END AS RSI_14_periods,
  -- MACD (5 periods)
  ema_12 - ema_26 AS MACD_5_periods,
  -- SMA (5 periods)
  AVG(Close) OVER (ORDER BY index ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS SMA_5_periods,
  -- SMA (15 periods)
  AVG(Close) OVER (ORDER BY index ROWS BETWEEN 14 PRECEDING AND CURRENT ROW) AS SMA_15_periods,
  bucket_number,
  COUNT(*) OVER (PARTITION BY bucket_number) AS bucket_count
FROM (
  SELECT
  Symbol,
    index,
    Close,
    AVG(Close) OVER (ORDER BY index ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS ema_12,
    AVG(Close) OVER (ORDER BY index ROWS BETWEEN 25 PRECEDING AND CURRENT ROW) AS ema_26,
    AVG(gain) OVER (ORDER BY index ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_gain,
    AVG(loss) OVER (ORDER BY index ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_loss,
    bucket_number
  FROM histogram_data
)
ORDER BY index DESC;
