SELECT
  timestamp,
  metric_name,
  column_name,
  column_type,
  stattest_name,
  stattest_threshold,
  drift_score,
  drift_detected
FROM
  `ambient-decoder-391319.evidently_report001.daily_report`
WHERE
  column_name <> "null"
  order by timestamp desc