# TYL Qdrant Adapter

🔍 **Qdrant vector database adapter for TYL framework with embedding integration and full vector operations**

This adapter provides a complete implementation of the TYL vector port for Qdrant, enabling seamless integration of Qdrant's high-performance vector database capabilities with the TYL hexagonal architecture.

[![Tests](https://github.com/the-yaml-life/tyl-qdrant-adapter/actions/workflows/ci.yml/badge.svg)](https://github.com/the-yaml-life/tyl-qdrant-adapter/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## ✨ Features

- 🔍 **Complete Qdrant Integration** - Full support for Qdrant vector database operations
- 🤖 **Embedding Service Integration** - Optional integration with TYL embedding services  
- 📊 **Collection Management** - Create, configure, and manage vector collections
- ⚡ **Batch Operations** - Efficient batch storage and retrieval
- 🏥 **Health Monitoring** - Built-in health checks and connection monitoring
- 🏗️ **TYL Framework Integration** - Uses TYL error handling, config, logging, and tracing
- 📈 **Observability** - Comprehensive logging and distributed tracing
- 🧪 **Mock Implementation** - In-memory mock for testing without Qdrant server

## 🚀 Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
tyl-qdrant-adapter = { git = "https://github.com/the-yaml-life/tyl-qdrant-adapter.git", branch = "main" }
```

### Basic Usage

```rust
use tyl_qdrant_adapter::{QdrantAdapter, QdrantConfig, ConfigPlugin};
use tyl_qdrant_adapter::{VectorDatabase, VectorStore, VectorCollectionManager};
use tyl_qdrant_adapter::{Vector, CollectionConfig, DistanceMetric, SearchParams};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure with environment variables
    let mut config = QdrantConfig::default();
    config.merge_env()?; // Loads TYL_QDRANT_* variables
    
    // Connect to Qdrant
    let adapter = QdrantAdapter::connect(config).await?;
    
    // Create a collection for embeddings
    let collection_config = CollectionConfig::new(
        "documents",
        768, // OpenAI embedding dimensions
        DistanceMetric::Cosine
    )?;
    adapter.create_collection(collection_config).await?;
    
    // Store a vector with metadata
    let vector = Vector::with_metadata(
        "doc_1".to_string(),
        vec![0.1; 768], // Your embedding vector
        std::collections::HashMap::from([
            ("title".to_string(), serde_json::json!("My Document")),
            ("category".to_string(), serde_json::json!("document"))
        ])
    );
    adapter.store_vector("documents", vector).await?;
    
    // Search for similar vectors
    let query_vector = vec![0.1; 768]; // Your query embedding
    let search_params = SearchParams::with_limit(5)
        .with_threshold(0.7)
        .with_filter("category", serde_json::json!("document"));
        
    let results = adapter.search_similar("documents", query_vector, search_params).await?;
    println!("Found {} similar documents", results.len());
    
    Ok(())
}
```

### Quick Setup with Docker

```bash
# Start Qdrant server
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

# Run the example
cargo run --example basic_usage
```

## ⚙️ Configuration

### Environment Variables

The adapter supports TYL configuration patterns with environment variable precedence:

| Variable | Default | Description |
|----------|---------|-------------|
| `TYL_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `TYL_QDRANT_API_KEY` | None | API key for authentication |
| `TYL_QDRANT_TIMEOUT_SECONDS` | `30` | Connection timeout |
| `TYL_QDRANT_MAX_BATCH_SIZE` | `100` | Maximum vectors per batch |
| `TYL_QDRANT_ENABLE_COMPRESSION` | `true` | Enable gRPC compression |
| `TYL_QDRANT_RETRY_ATTEMPTS` | `3` | Failed operation retries |
| `TYL_QDRANT_RETRY_DELAY_MS` | `1000` | Delay between retries |

### Programmatic Configuration

```rust
use tyl_qdrant_adapter::{QdrantConfig, ConfigPlugin};

let config = QdrantConfig {
    url: "http://localhost:6333".to_string(),
    api_key: Some("your-api-key".to_string()),
    timeout_seconds: 30,
    max_batch_size: 100,
    enable_compression: true,
    retry_attempts: 3,
    retry_delay_ms: 1000,
    default_shard_number: 1,
    default_replication_factor: 1,
};

// Validate configuration
config.validate()?;
```

## 🏗️ Architecture

This adapter implements the TYL vector port traits following hexagonal architecture:

```rust
// Core interfaces
trait VectorStore {
    async fn store_vector(&self, collection: &str, vector: Vector) -> TylResult<()>;
    async fn search_similar(&self, collection: &str, query: Vec<f32>, params: SearchParams) -> TylResult<Vec<VectorSearchResult>>;
    // ... more operations
}

trait VectorCollectionManager {
    async fn create_collection(&self, config: CollectionConfig) -> TylResult<()>;
    async fn list_collections(&self) -> TylResult<Vec<CollectionConfig>>;
    // ... more operations  
}

trait VectorStoreHealth {
    async fn is_healthy(&self) -> TylResult<bool>;
    async fn health_check(&self) -> TylResult<HashMap<String, serde_json::Value>>;
}
```

### Implementations

- **`QdrantAdapter`** - Production implementation using Qdrant gRPC client
- **`MockQdrantAdapter`** - In-memory mock for testing without Qdrant server

## 🧪 Testing

### Running Tests

```bash
# Run all tests (recommended - uses MockQdrantAdapter)
cargo test

# Run integration tests with mock adapter
cargo test --test integration_tests

# Test documentation examples
cargo test --doc

# Docker integration testing (experimental)
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
cargo run --example docker_testing --features docker-testing
```

**Docker Integration**: Real Qdrant integration tests are now fully functional with qdrant-client v1.15.0 and Qdrant server v1.15.3. The MockQdrantAdapter provides fast testing without dependencies, while Docker tests validate real server behavior.

### Testing with Mock Adapter

```rust
use tyl_qdrant_adapter::MockQdrantAdapter;
use tyl_qdrant_adapter::{VectorStore, CollectionConfig, Vector, DistanceMetric};

#[tokio::test]
async fn test_vector_operations() {
    let adapter = MockQdrantAdapter::new();
    
    // Create collection
    let config = CollectionConfig::new("test", 128, DistanceMetric::Cosine)?;
    adapter.create_collection(config).await?;
    
    // Store and retrieve vector
    let vector = Vector::new("test_id".to_string(), vec![0.1; 128]);
    adapter.store_vector("test", vector).await?;
    
    let retrieved = adapter.get_vector("test", "test_id").await?;
    assert!(retrieved.is_some());
}
```

## 📊 Monitoring & Observability

The adapter provides comprehensive observability through TYL logging and tracing:

### Structured Logging

All operations are logged with structured JSON:

```json
{
  "level": "INFO",
  "message": "qdrant_store_vector - Storing vector 'doc_1' in collection 'documents'",
  "timestamp": "2024-01-15T10:30:45Z"
}
```

### Distributed Tracing

Operations create spans for distributed tracing:

- `qdrant_store_vector` - Vector storage operations
- `qdrant_search_similar` - Similarity search operations  
- `qdrant_create_collection` - Collection creation operations

### Health Monitoring

```rust
// Check if Qdrant is healthy
let healthy = adapter.is_healthy().await?;

// Get detailed health information
let health_data = adapter.health_check().await?;
println!("Qdrant status: {}", health_data["status"]);
```

## 🔄 Schema Migration & Contract Testing

The adapter provides sophisticated schema migration tools with Pact.io validation for production deployments:

### Features

- ✅ **Semantic Versioning** - Track collection schema changes with semver
- ✅ **Dependency Management** - Define migration dependencies and order
- ✅ **Pact.io Integration** - Contract-based testing for microservice compatibility
- ✅ **Rollback Support** - Safe rollback of reversible migrations
- ✅ **Breaking Change Detection** - Automatic identification of breaking changes
- ✅ **Migration History** - Complete audit trail of schema changes

### Enable Migration Features

Add to your `Cargo.toml`:

```toml
[dependencies]
tyl-qdrant-adapter = { 
    git = "https://github.com/the-yaml-life/tyl-qdrant-adapter.git", 
    branch = "main",
    features = ["schema-migration"]
}
```

### Usage

```rust
use tyl_qdrant_adapter::{QdrantAdapter, migration::*};
use semver::Version;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = QdrantAdapter::connect(config).await?;
    let manager = SchemaMigrationManager::new(adapter);
    
    // Initialize migration system
    manager.initialize().await?;
    
    // Create migration with Pact contract validation
    let pact_contract = PactContract {
        consumer: "document-service".to_string(),
        provider: "qdrant-adapter".to_string(),
        contract_path: "./pacts/document-service-qdrant-adapter.json".to_string(),
        interactions: vec![
            PactInteraction {
                description: "create documents collection".to_string(),
                request: VectorRequest {
                    operation: VectorOperation::CreateCollection,
                    collection: "documents".to_string(),
                    parameters: serde_json::json!({
                        "dimension": 768,
                        "distance_metric": "Cosine"
                    }),
                },
                response: VectorResponse {
                    status: ResponseStatus::Success,
                    data: Some(serde_json::json!({"created": true})),
                    error: None,
                },
            }
        ],
    };
    
    // Build migration with fluent API
    let migration = MigrationBuilder::new(
        Version::new(1, 0, 0), 
        "Add documents collection".to_string()
    )
    .author("DevOps Team".to_string())
    .description("Create initial documents collection with 768-dim embeddings")
    .create_collection(
        CollectionConfig::new("documents", 768, DistanceMetric::Cosine)?
    )
    .add_pact_contract(pact_contract)
    .build();
    
    // Apply migration with contract validation
    let result = manager.apply_migration(migration).await?;
    println!("Migration {} applied successfully", result.version);
    
    Ok(())
}
```

### Advanced Migration Patterns

```rust
// Complex migration with dependencies
let migration = MigrationBuilder::new(Version::new(2, 0, 0), "Major schema update".to_string())
    .author("Platform Team".to_string())
    .description("Update to support multi-modal embeddings")
    .depends_on(Version::new(1, 0, 0)) // Requires previous migration
    .breaking_change() // Mark as breaking
    .create_collection(
        CollectionConfig::new("images", 512, DistanceMetric::Cosine)?
    )
    .create_collection(
        CollectionConfig::new("audio", 256, DistanceMetric::Euclidean)?
    )
    .build();

// Advanced filtering with complex conditions
let must_conditions = vec![
    ("category".to_string(), serde_json::json!("document")),
    ("status".to_string(), serde_json::json!("published")),
];

let should_conditions = vec![
    ("priority".to_string(), serde_json::json!("high")),
    ("priority".to_string(), serde_json::json!("urgent")),
];

let filter = QdrantAdapter::build_complex_filter(
    must_conditions,
    should_conditions,
    vec![], // no must_not conditions
)?;

// Range filtering for numeric fields
let range_filter = QdrantAdapter::build_range_filter(
    "created_timestamp",
    Some(1640995200.0), // min timestamp
    Some(1672531200.0), // max timestamp
)?;
```

### Migration Management

```rust
// Get migration history
let history = manager.get_migration_history().await?;
for migration in &history {
    println!("✅ {} - {} ({})", 
        migration.version, 
        migration.name,
        migration.metadata.author
    );
}

// Rollback to previous version (if reversible)
manager.rollback_migration(Version::new(2, 0, 0)).await?;

// Check migration status
if let Some(latest) = history.last() {
    println!("Current schema version: {}", latest.version);
}
```

### Contract Testing Integration

The migration system integrates with Pact.io for microservice compatibility:

1. **Consumer Contracts** - Define expected interactions with vector store
2. **Provider Validation** - Validate adapter behavior against contracts  
3. **Breaking Change Detection** - Prevent incompatible schema changes
4. **CI/CD Integration** - Automated contract validation in pipelines

```yaml
# .github/workflows/migration.yml
name: Schema Migration
on:
  push:
    paths:
      - 'migrations/**'

jobs:
  validate-migration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run migration with Pact validation
        run: |
          cargo test --features schema-migration --test migration_tests
          pact-broker publish ./pacts --consumer-app-version=${{ github.sha }}
```

## 🔧 Error Handling

The adapter provides domain-specific error helpers following TYL patterns:

```rust
use tyl_qdrant_adapter::qdrant_errors;

// Specific error scenarios
let conn_error = qdrant_errors::connection_failed("Network timeout");
let dim_error = qdrant_errors::vector_dimension_mismatch(768, 512);
let batch_error = qdrant_errors::batch_size_exceeded(1000, 100);
let collection_error = qdrant_errors::collection_creation_failed("docs", "Permission denied");
```

## 🎯 Production Considerations

### Performance

- **Batch Operations** - Use `store_vectors_batch()` for bulk operations
- **Connection Pooling** - Qdrant client handles connection pooling internally  
- **Compression** - gRPC compression enabled by default
- **Timeouts** - Configurable timeouts prevent hanging operations

### Security

- **API Keys** - Support for Qdrant API key authentication
- **Input Validation** - All inputs are validated before processing
- **Error Sanitization** - Errors don't leak sensitive information

### Monitoring

- **Health Checks** - Built-in health monitoring for service discovery
- **Metrics** - Operation timing and success/failure rates via logging
- **Tracing** - End-to-end request tracing for debugging

## 🔗 Integration with TYL Framework

This adapter integrates seamlessly with other TYL framework components:

- **[tyl-errors](https://github.com/the-yaml-life/tyl-errors)** - Comprehensive error handling
- **[tyl-config](https://github.com/the-yaml-life/tyl-config)** - Configuration management
- **[tyl-logging](https://github.com/the-yaml-life/tyl-logging)** - Structured logging
- **[tyl-tracing](https://github.com/the-yaml-life/tyl-tracing)** - Distributed tracing
- **[tyl-vector-port](https://github.com/the-yaml-life/tyl-vector-port)** - Vector operations interface
- **[tyl-embeddings-port](https://github.com/the-yaml-life/tyl-embeddings-port)** - Embedding services

## 📚 Examples

### Basic Vector Operations

See [`examples/basic_usage.rs`](examples/basic_usage.rs) for a comprehensive example covering:

- Configuration with environment variables
- Collection creation and management
- Vector storage with metadata
- Similarity search with filtering
- Error handling and logging
- Health monitoring

### Advanced Usage

```rust
// Batch operations for performance
let vectors = vec![
    Vector::new("doc1".to_string(), vec![0.1; 768]),
    Vector::new("doc2".to_string(), vec![0.2; 768]),
    Vector::new("doc3".to_string(), vec![0.3; 768]),
];

let results = adapter.store_vectors_batch("documents", vectors).await?;
for (i, result) in results.iter().enumerate() {
    match result {
        Ok(_) => println!("Vector {} stored successfully", i),
        Err(e) => println!("Vector {} failed: {}", i, e),
    }
}

// Advanced search with metadata filtering
let search_params = SearchParams::with_limit(10)
    .with_threshold(0.8)
    .with_filter("category", serde_json::json!("technical"))
    .with_filter("language", serde_json::json!("en"));

let results = adapter.search_similar("documents", query_vector, search_params).await?;
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow TDD approach - write tests first
4. Ensure all tests pass (`cargo test`)
5. Check code quality (`cargo clippy`, `cargo fmt`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📝 License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Issues** - [GitHub Issues](https://github.com/the-yaml-life/tyl-qdrant-adapter/issues)
- **Discussions** - [GitHub Discussions](https://github.com/the-yaml-life/tyl-qdrant-adapter/discussions)
- **Documentation** - [CLAUDE.md](CLAUDE.md) for detailed technical documentation