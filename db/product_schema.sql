CREATE EXTENSION IF NOT EXISTS vector;
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
  embedding vector(1536)  -- adjust to your embedding model dimension if needed
);

CREATE INDEX products_category_idx ON products (category);
CREATE INDEX products_color_idx ON products (color);
CREATE INDEX products_price_idx ON products (price);
