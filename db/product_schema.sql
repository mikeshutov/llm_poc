DROP TABLE IF EXISTS products;

CREATE TABLE products (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  category TEXT,
  color TEXT,
  style TEXT,
  gender TEXT,
  season TEXT,
  year INT,
  price NUMERIC,
  image_url TEXT,
  embedding vector(1536)
);

CREATE INDEX products_category_idx ON products (category);
CREATE INDEX products_color_idx ON products (color);
CREATE INDEX products_price_idx ON products (price);
CREATE INDEX IF NOT EXISTS products_embedding_idx ON products USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
