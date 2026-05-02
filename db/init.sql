CREATE EXTENSION IF NOT EXISTS vector;

DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'mcp_agent') THEN

      CREATE ROLE mcp_agent LOGIN PASSWORD 'mcp_secure_password';
   END IF;
END
$do$;

GRANT USAGE ON SCHEMA public TO mcp_agent;

GRANT SELECT ON ALL TABLES IN SCHEMA public TO mcp_agent;

ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO mcp_agent;