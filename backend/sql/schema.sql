-- Edital na mão: schema do Postgres (Supabase ou qualquer Postgres com pgvector)
-- A dimensão do vetor precisa bater com EMBED_DIM (padrão 768).

create extension if not exists vector;

create table if not exists editais (
  id            text primary key,
  titulo        text not null,
  paginas       int  not null,
  total_trechos int  not null default 0,
  fonte_url     text,
  criado_em     timestamptz not null default now()
);

create table if not exists trechos (
  id         text primary key,          -- "<edital_id>#<ordem>"
  edital_id  text not null references editais(id) on delete cascade,
  ordem      int  not null,
  pagina     int  not null,
  secao      text,
  texto      text not null,
  embedding  vector(768) not null,
  -- Full text em português (com stemming): "inscrições" acha "inscrição".
  tsv        tsvector generated always as (
               to_tsvector('portuguese', coalesce(secao, '') || ' ' || texto)
             ) stored
);

create index if not exists trechos_edital_idx on trechos (edital_id);
create index if not exists trechos_tsv_idx on trechos using gin (tsv);
create index if not exists trechos_embedding_idx on trechos using hnsw (embedding vector_cosine_ops);

-- A API acessa com a connection string do banco (role postgres), não pela API
-- pública do Supabase. RLS ligado sem política = ninguém de fora lê via REST.
alter table editais enable row level security;
alter table trechos enable row level security;
