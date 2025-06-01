CREATE EXTENSION vector;
ALTER DATABASE crossretrieval SET timezone TO 'Europe/Helsinki';

create TABLE if NOT EXISTS dim_item_types(
  item_type_id  INT       GENERATED ALWAYS AS IDENTITY
  , item_type   TEXT
  , PRIMARY KEY (item_type_id)
);

CREATE TABLE items (
  id            INT       GENERATED ALWAYS AS IDENTITY
  ,item_type    INT
  ,location     TEXT
  ,embedding    vector(512)
  ,dt_added     timestamp DEFAULT CURRENT_TIMESTAMP

  , PRIMARY KEY (id)
  , CONSTRAINT fk_dim_item_types FOREIGN KEY (item_type) REFERENCES dim_item_types(item_type_id) ON DELETE SET NULL
);

CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);
