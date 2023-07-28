WITH data AS (
  SELECT
  timestamp, 
    SPLIT(REGEXP_REPLACE(current_x, r'[\[\]]', ''), ',') AS current_x_array,
    SPLIT(REGEXP_REPLACE(current_y, r'[\[\]]', ''), ',') AS current_y_array,
    SPLIT(REGEXP_REPLACE(reference_x, r'[\[\]]', ''), ',') AS reference_x_array,
    SPLIT(REGEXP_REPLACE(reference_y, r'[\[\]]', ''), ',') AS reference_y_array
  FROM `ambient-decoder-391319.evidently_report001.daily_report`
)
SELECT
timestamp,
  CAST(ROUND(CAST(x AS NUMERIC), 6) AS NUMERIC) AS current_x,
  CAST(ROUND(CAST(y AS NUMERIC), 6) AS NUMERIC) AS current_y,
  CAST(ROUND(CAST(z AS NUMERIC), 6) AS NUMERIC) AS reference_x,
  CAST(ROUND(CAST(w AS NUMERIC), 6) AS NUMERIC) AS reference_y
FROM data,
UNNEST(current_x_array) AS x WITH OFFSET x_offset
JOIN UNNEST(current_y_array) AS y WITH OFFSET y_offset
JOIN UNNEST(reference_x_array) AS z WITH OFFSET z_offset
JOIN UNNEST(reference_y_array) AS w WITH OFFSET w_offset
WHERE x_offset = y_offset
  AND x_offset = z_offset
  AND x_offset = w_offset
order by timestamp desc limit 10
