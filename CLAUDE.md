# CLAUDE.md - tyl-qdrant-adapter

## 📋 **Module Context**

**tyl-qdrant-adapter** is the Qdrant vector database adapter module for the TYL framework. It provides a complete implementation of the TYL vector port for Qdrant, enabling seamless integration of Qdrant's high-performance vector database capabilities with the TYL hexagonal architecture.

## 🏗️ **Architecture**

### **Port (Interface)**
```rust
// Vector Operations
trait VectorStore {
    async fn store_vector(&self, collection: &str, vector: Vector) -> TylResult<()>;
    async fn get_vector(&self, collection: &str, id: &str) -> TylResult<Option<Vector>>;
    async fn search_similar(&self, collection: &str, query: Vec<f32>, params: SearchParams) -> TylResult<Vec<VectorSearchResult>>;
    async fn delete_vector(&self, collection: &str, id: &str) -> TylResult<()>;
}

// Collection Management
trait VectorCollectionManager {
    async fn create_collection(&self, config: CollectionConfig) -> TylResult<()>;
    async fn delete_collection(&self, collection_name: &str) -> TylResult<()>;
    async fn list_collections(&self) -> TylResult<Vec<CollectionConfig>>;
}

// Health Monitoring
trait VectorStoreHealth {
    async fn is_healthy(&self) -> TylResult<bool>;
    async fn health_check(&self) -> TylResult<HashMap<String, serde_json::Value>>;
}
```

### **Adapters (Implementations)**
- `QdrantAdapter` - Production implementation using Qdrant gRPC client
- `MockQdrantAdapter` - In-memory mock implementation for testing

### **Core Types**
- `QdrantConfig` - Configuration with TYL patterns
- `Vector` - Domain vector type with metadata
- `CollectionConfig` - Collection specification with distance metrics
- `SearchParams` - Search parameters with filtering capabilities

## 🧪 **Testing**

```bash
# Run all tests (uses MockQdrantAdapter - reliable)
cargo test -p tyl-qdrant-adapter

# Run integration tests with mock adapter (recommended)
cargo test --test integration_tests -p tyl-qdrant-adapter

# Run documentation tests
cargo test --doc -p tyl-qdrant-adapter

# Run basic usage example
cargo run --example basic_usage -p tyl-qdrant-adapter

# Docker integration testing (experimental - has compatibility issues)
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
cargo run --example docker_testing --features docker-testing -p tyl-qdrant-adapter
```

### **Docker Integration Testing**

✅ **Real Qdrant Integration**: Full Docker integration tests are now working with qdrant-client v1.15.0 and Qdrant server v1.15.3.

**Successful Testing Setup**:
- **Mock Testing**: `MockQdrantAdapter` for unit tests (fast, no dependencies)
- **Integration Testing**: `integration_tests.rs` for comprehensive mock-based testing  
- **Docker Testing**: `docker_integration_tests.rs` for real Qdrant server validation

**Key Docker Integration Insights**:
- Use gRPC port (6334) instead of HTTP port (6333)
- Vector IDs must be UUID format for compatibility
- Qdrant normalizes vectors automatically (cosine distance)
- Enable `include_vectors()` in SearchParams for proper vector retrieval

## 📂 **File Structure**

```
tyl-qdrant-adapter/
├── src/
│   ├── lib.rs                 # Core implementation with logging & tracing
│   └── mock.rs                # Mock implementation for testing
├── examples/
│   └── basic_usage.rs         # Comprehensive usage example
├── tests/
│   └── integration_tests.rs   # Integration tests with mock adapter
├── README.md                  # Main documentation
├── CLAUDE.md                  # This file
└── Cargo.toml                 # Package metadata with TYL dependencies
```

## 🔧 **How to Use**

### **Basic Usage**
```rust
use tyl_qdrant_adapter::{QdrantAdapter, QdrantConfig, ConfigPlugin};
use tyl_qdrant_adapter::{VectorDatabase, VectorStore, VectorCollectionManager};
use tyl_qdrant_adapter::{Vector, CollectionConfig, DistanceMetric, SearchParams};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure adapter with TYL patterns
    let mut config = QdrantConfig::default();
    config.merge_env()?; // Loads from TYL_QDRANT_* environment variables
    
    // Connect to Qdrant
    let adapter = QdrantAdapter::connect(config).await?;
    
    // Create vector collection
    let collection_config = CollectionConfig::new(
        "documents",
        768, // OpenAI embedding dimensions
        DistanceMetric::Cosine
    )?;
    adapter.create_collection(collection_config).await?;
    
    // Store vectors with metadata
    let vector = Vector::with_metadata(
        "doc_1".to_string(),
        vec![0.1; 768],
        HashMap::from([
            ("title".to_string(), serde_json::json!("Document Title")),
            ("category".to_string(), serde_json::json!("document"))
        ])
    );
    adapter.store_vector("documents", vector).await?;
    
    // Search similar vectors
    let query_vector = vec![0.1; 768];
    let search_params = SearchParams::with_limit(5)
        .with_threshold(0.7)
        .with_filter("category", serde_json::json!("document"));
        
    let results = adapter.search_similar("documents", query_vector, search_params).await?;
    
    Ok(())
}
```

### **Configuration Management**
```rust
use tyl_qdrant_adapter::{QdrantConfig, ConfigPlugin};

// TYL configuration patterns with validation
let mut config = QdrantConfig::default();

// Automatic environment loading with TYL precedence:
// TYL_QDRANT_URL > QDRANT_URL > default
// TYL_QDRANT_API_KEY > QDRANT_API_KEY > default
config.merge_env()?;

// Validation using TYL patterns
config.validate()?;

println!("Qdrant URL: {}", config.url);
println!("Timeout: {}s", config.timeout_seconds);
println!("Batch Size: {}", config.max_batch_size);
```

### **Error Handling with TYL Patterns**
```rust
use tyl_qdrant_adapter::qdrant_errors;

// Domain-specific error helpers
let conn_error = qdrant_errors::connection_failed("Network timeout");
let dim_error = qdrant_errors::vector_dimension_mismatch(768, 512);
let batch_error = qdrant_errors::batch_size_exceeded(1000, 100);
let collection_error = qdrant_errors::collection_creation_failed("docs", "Permission denied");
```

### **Testing with Mock Adapter**
```rust
use tyl_qdrant_adapter::MockQdrantAdapter;

#[tokio::test]
async fn test_vector_operations() {
    let adapter = MockQdrantAdapter::new();
    
    // Test without requiring real Qdrant instance
    let config = CollectionConfig::new("test", 128, DistanceMetric::Cosine)?;
    adapter.create_collection(config).await?;
    
    let vector = Vector::new("test_id".to_string(), vec![0.1; 128]);
    adapter.store_vector("test", vector).await?;
    
    let retrieved = adapter.get_vector("test", "test_id").await?;
    assert!(retrieved.is_some());
}
```

## 🛠️ **Useful Commands**

```bash
# Code quality
cargo clippy -p tyl-qdrant-adapter -- -D warnings
cargo fmt -p tyl-qdrant-adapter  
cargo doc --no-deps -p tyl-qdrant-adapter --open

# Testing
cargo test -p tyl-qdrant-adapter --verbose
cargo test --doc -p tyl-qdrant-adapter

# Start Qdrant for local development
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

# Example with real Qdrant
TYL_QDRANT_URL=http://localhost:6333 \
cargo run --example basic_usage -p tyl-qdrant-adapter
```

## 📦 **Dependencies**

### **TYL Framework Dependencies**
- `tyl-errors` - Comprehensive error handling with TYL patterns
- `tyl-config` - Configuration management with environment precedence
- `tyl-logging` - Structured logging with JSON output
- `tyl-tracing` - Distributed tracing and observability
- `tyl-vector-port` - Vector operations port definition
- `tyl-embeddings-port` - Embedding services integration

### **Qdrant Integration**
- `qdrant-client` - Official Qdrant gRPC client (v1.15+)
- `reqwest` - HTTP client for Qdrant REST API fallback

### **Core Dependencies**
- `async-trait` - Async trait support
- `serde` - Serialization with derive support
- `tokio` - Async runtime with full features
- `uuid` - Vector ID generation

## 🎯 **Design Principles**

1. **Hexagonal Architecture** - Clean separation between domain logic and Qdrant specifics
2. **TYL Framework Integration** - Full use of TYL error handling, config, logging, and tracing
3. **Type Safety** - Strong type conversions between TYL and Qdrant domains
4. **Performance Focus** - Efficient batch operations and connection management
5. **Observability** - Comprehensive logging and tracing for production monitoring
6. **Testing Strategy** - Mock adapter for unit tests, real adapter for integration tests

## ⚙️ **Configuration Options**

### **Connection Settings**
- `url` - Qdrant server URL (default: http://localhost:6333)
- `api_key` - Authentication API key (optional for local instances)
- `timeout_seconds` - Connection timeout (default: 30s)
- `enable_compression` - gRPC compression (default: true)

### **Performance Settings**
- `max_batch_size` - Maximum vectors per batch (default: 100)
- `retry_attempts` - Failed operation retries (default: 3)
- `retry_delay_ms` - Delay between retries (default: 1000ms)

### **Collection Defaults**
- `default_shard_number` - Shards per collection (default: 1)
- `default_replication_factor` - Replication factor (default: 1)

## 🚀 **Production Considerations**

### **Logging and Monitoring**
- All operations logged with structured JSON logging
- Distributed tracing for performance monitoring
- Health check endpoints for service monitoring
- Collection statistics for capacity planning

### **Error Handling**
- Comprehensive error categorization (network, validation, database, performance)
- Retry logic for transient failures
- Graceful degradation for connection issues
- Detailed error context for debugging

### **Performance Optimization**
- Batch operations for bulk vector storage
- Connection pooling with health monitoring
- Configurable timeouts and retry policies
- Efficient type conversions between domains

## ⚠️ **Known Limitations**

- **Filter Support**: Limited filter implementation (TODO: enhance filtering capabilities)
- **Collection Migration**: No automated schema migration tools
- **Index Optimization**: Manual index configuration required
- **Backup/Restore**: No built-in backup utilities

## 📝 **Notes for Contributors**

- Follow TDD approach with comprehensive test coverage
- Maintain hexagonal architecture separation
- Use TYL error helpers for domain-specific errors
- Add logging and tracing to all major operations
- Document all public APIs with usage examples
- Keep Qdrant client API compatibility up to date

## 🔗 **Related TYL Modules**

- [`tyl-errors`](https://github.com/the-yaml-life/tyl-errors) - Error handling patterns
- [`tyl-config`](https://github.com/the-yaml-life/tyl-config) - Configuration management
- [`tyl-logging`](https://github.com/the-yaml-life/tyl-logging) - Structured logging
- [`tyl-tracing`](https://github.com/the-yaml-life/tyl-tracing) - Distributed tracing
- [`tyl-vector-port`](https://github.com/the-yaml-life/tyl-vector-port) - Vector operations port
- [`tyl-embeddings-port`](https://github.com/the-yaml-life/tyl-embeddings-port) - Embedding services
- [`tyl-ollama-embedding-adapter`](https://github.com/the-yaml-life/tyl-ollama-embedding-adapter) - Ollama embeddings

## 🏆 **Quality Standards**

This module follows TYL framework quality standards:

- ✅ **TYL Integration**: Full framework integration with error handling, config, logging, tracing
- ✅ **Architecture**: Hexagonal architecture with proper port/adapter separation  
- ✅ **Error Handling**: Domain-specific error helpers with TYL categorization
- ✅ **Testing**: Comprehensive unit and integration tests with mock adapter
- ✅ **Documentation**: Complete API documentation with usage examples
- ✅ **Performance**: Efficient operations with logging and tracing
- ✅ **Configuration**: TYL patterns with environment precedence and validation