-- Initialize PostGIS and pgvector extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

-- Sessions
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT now(),
    settings JSONB DEFAULT '{}'
);

-- Layers
CREATE TABLE IF NOT EXISTS layers (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE NOT NULL,
    name VARCHAR(512) NOT NULL,
    layer_type VARCHAR(50) NOT NULL,
    metadata JSONB DEFAULT '{}',
    storage_path VARCHAR(1024),
    dataframe_json TEXT,
    geodata_json TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Text chunks with pgvector embeddings
CREATE TABLE IF NOT EXISTS text_chunks (
    id SERIAL PRIMARY KEY,
    layer_id INTEGER REFERENCES layers(id) ON DELETE CASCADE NOT NULL,
    source VARCHAR(512) NOT NULL,
    page INTEGER DEFAULT 1,
    text TEXT NOT NULL,
    embedding vector(384)
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON text_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Analysis outputs
CREATE TABLE IF NOT EXISTS outputs (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE NOT NULL,
    title VARCHAR(512) NOT NULL,
    output_type VARCHAR(50) NOT NULL,
    content_json TEXT,
    figure_path VARCHAR(1024),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Chat messages
CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Spatial features (PostGIS)
CREATE TABLE IF NOT EXISTS layer_features (
    id SERIAL PRIMARY KEY,
    layer_id INTEGER REFERENCES layers(id) ON DELETE CASCADE NOT NULL,
    geom GEOMETRY(Geometry, 4326),
    properties JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_features_geom
    ON layer_features USING GIST (geom);
