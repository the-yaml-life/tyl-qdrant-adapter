//! Integration tests for TYL Qdrant Adapter
//!
//! These tests verify the integration between the Qdrant adapter and the TYL framework,
//! including vector operations, embedding services, and configuration management.

use tyl_qdrant_adapter::{
    CollectionConfig, ConfigPlugin, DistanceMetric, MockQdrantAdapter, QdrantConfig, SearchParams,
    Vector, VectorCollectionManager, VectorDatabase, VectorStore, VectorStoreHealth,
};

#[tokio::test]
async fn test_mock_adapter_integration() {
    let adapter = MockQdrantAdapter::new();

    // Test collection creation
    let collection_config = CollectionConfig::new("test_collection", 128, DistanceMetric::Cosine)
        .expect("Failed to create collection config");

    let result = adapter.create_collection(collection_config).await;
    assert!(result.is_ok());

    // Test vector storage
    let vector = Vector::new("test_id".to_string(), vec![0.1; 128]);
    let result = adapter.store_vector("test_collection", vector).await;
    assert!(result.is_ok());

    // Test vector retrieval
    let result = adapter.get_vector("test_collection", "test_id").await;
    assert!(result.is_ok());
    assert!(result.unwrap().is_some());
}

#[tokio::test]
async fn test_configuration_integration() {
    let mut config = QdrantConfig::default();
    assert!(config.validate().is_ok());

    // Test configuration loading
    std::env::set_var("TYL_QDRANT_URL", "http://test:6333");
    std::env::set_var("TYL_QDRANT_TIMEOUT_SECONDS", "60");

    let result = config.merge_env();
    assert!(result.is_ok());
    assert_eq!(config.url, "http://test:6333");
    assert_eq!(config.timeout_seconds, 60);

    // Cleanup
    std::env::remove_var("TYL_QDRANT_URL");
    std::env::remove_var("TYL_QDRANT_TIMEOUT_SECONDS");
}

#[tokio::test]
async fn test_vector_operations_integration() {
    let adapter = MockQdrantAdapter::new();

    // Create collection
    let collection_config = CollectionConfig::new("vectors_test", 256, DistanceMetric::Euclidean)
        .expect("Failed to create collection config");

    let result = adapter.create_collection(collection_config).await;
    assert!(result.is_ok());

    // Store multiple vectors
    let vectors = vec![
        Vector::new("vec1".to_string(), vec![0.1; 256]),
        Vector::new("vec2".to_string(), vec![0.2; 256]),
        Vector::new("vec3".to_string(), vec![0.3; 256]),
    ];

    let results = adapter.store_vectors_batch("vectors_test", vectors).await;
    assert!(results.is_ok());
    let batch_results = results.unwrap();
    assert_eq!(batch_results.len(), 3);
    assert!(batch_results.iter().all(|r| r.is_ok()));

    // Test search
    let query_vector = vec![0.15; 256];
    let search_params = SearchParams::with_limit(2);

    let search_results = adapter
        .search_similar("vectors_test", query_vector, search_params)
        .await;
    assert!(search_results.is_ok());
    let results = search_results.unwrap();
    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_health_check_integration() {
    let adapter = MockQdrantAdapter::new();

    // Test health check
    let is_healthy = adapter.is_healthy().await;
    assert!(is_healthy.is_ok());
    assert!(is_healthy.unwrap());

    // Test detailed health check
    let health_data = adapter.health_check().await;
    assert!(health_data.is_ok());
    let data = health_data.unwrap();
    assert!(data.contains_key("status"));
    assert_eq!(data["status"], serde_json::json!("healthy"));
}

#[tokio::test]
async fn test_collection_management_integration() {
    let adapter = MockQdrantAdapter::new();

    // Create multiple collections
    let configs = vec![
        CollectionConfig::new("collection1", 128, DistanceMetric::Cosine).unwrap(),
        CollectionConfig::new("collection2", 256, DistanceMetric::Euclidean).unwrap(),
        CollectionConfig::new("collection3", 512, DistanceMetric::DotProduct).unwrap(),
    ];

    for config in configs {
        let result = adapter.create_collection(config).await;
        assert!(result.is_ok());
    }

    // List collections
    let collections = adapter.list_collections().await;
    assert!(collections.is_ok());
    let collection_list = collections.unwrap();
    assert_eq!(collection_list.len(), 3);

    // Get collection info
    let info = adapter.get_collection_info("collection1").await;
    assert!(info.is_ok());
    assert!(info.unwrap().is_some());

    // Get collection stats
    let stats = adapter.get_collection_stats("collection1").await;
    assert!(stats.is_ok());
    let stats_data = stats.unwrap();
    assert!(stats_data.contains_key("vectors_count"));

    // Delete collection
    let result = adapter.delete_collection("collection1").await;
    assert!(result.is_ok());

    // Verify deletion
    let info = adapter.get_collection_info("collection1").await;
    assert!(info.is_ok());
    assert!(info.unwrap().is_none());
}

#[tokio::test]
async fn test_vector_database_trait_integration() {
    // Test mock database connection
    let config = QdrantConfig::default();
    let adapter = MockQdrantAdapter::connect(config).await;
    assert!(adapter.is_ok());

    let adapter = adapter.unwrap();

    // Test connection info
    let info = adapter.connection_info();
    assert!(info.contains("Mock"));

    // Test feature support
    assert!(adapter.supports_feature("collections"));
    assert!(adapter.supports_feature("health_check"));
    assert!(adapter.supports_feature("batch_operations"));
    assert!(!adapter.supports_feature("non_existent_feature"));

    // Test close connection
    let mut adapter = adapter;
    let result = adapter.close().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_error_handling_integration() {
    let adapter = MockQdrantAdapter::new();

    // Test getting non-existent vector
    let result = adapter
        .get_vector("non_existent_collection", "non_existent_id")
        .await;
    assert!(result.is_err());

    // Test searching in non-existent collection
    let search_params = SearchParams::with_limit(10);
    let result = adapter
        .search_similar("non_existent_collection", vec![0.1; 128], search_params)
        .await;
    assert!(result.is_err());

    // Test deleting from non-existent collection
    let result = adapter
        .delete_vector("non_existent_collection", "some_id")
        .await;
    assert!(result.is_err());

    // Test duplicate collection creation
    let config = CollectionConfig::new("test_dup", 128, DistanceMetric::Cosine).unwrap();
    let result1 = adapter.create_collection(config.clone()).await;
    assert!(result1.is_ok());

    let result2 = adapter.create_collection(config).await;
    assert!(result2.is_err());
}

#[test]
fn test_qdrant_config_serialization() {
    let config = QdrantConfig::default();

    // Test JSON serialization
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: QdrantConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config.url, deserialized.url);
    assert_eq!(config.timeout_seconds, deserialized.timeout_seconds);
    assert_eq!(config.max_batch_size, deserialized.max_batch_size);
}
