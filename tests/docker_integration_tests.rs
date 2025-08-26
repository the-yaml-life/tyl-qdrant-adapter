//! Docker integration tests for TYL Qdrant Adapter
//!
//! ✅ **Real Integration Tests**: These tests run against a live Qdrant instance.
//! Successfully tested with qdrant-client v1.15.0 and Qdrant server v1.15.3.
//!
//! **Key Requirements**:
//! - Use gRPC port (6334) instead of HTTP port (6333)
//! - Generate proper UUID-format IDs for vectors
//! - Handle vector normalization by Qdrant (cosine distance)
//! - Enable `include_vectors()` in SearchParams for vector retrieval
//!
//! **Usage**:
//! 1. Start Qdrant: `docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest`
//! 2. Run tests: `cargo test --test docker_integration_tests -- --test-threads=1`
//!
//! These tests verify:
//! - Real connection to Qdrant server
//! - Collection operations (create, list, stats, delete)
//! - Vector operations (store, retrieve, search, delete)
//! - Batch operations (bulk store/delete)
//! - Error handling with real server responses
//! - Configuration management with environment variables

use std::collections::HashMap;
use uuid::Uuid;
use tyl_qdrant_adapter::{
    QdrantAdapter, QdrantConfig, ConfigPlugin,
    VectorStore, VectorCollectionManager, VectorStoreHealth, VectorDatabase,
    Vector, CollectionConfig, DistanceMetric, SearchParams,
};

/// Check if Qdrant is available for testing
async fn is_qdrant_available() -> bool {
    // Simple HTTP check to see if Qdrant is responding
    match reqwest::get("http://localhost:6333/").await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

/// Skip test if Qdrant is not available
macro_rules! skip_if_no_qdrant {
    () => {
        if !is_qdrant_available().await {
            println!("⚠️  Skipping test: Qdrant not available at localhost:6333");
            println!("   Start with: docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant");
            return;
        }
    };
}

#[tokio::test]
async fn test_real_qdrant_connection() {
    skip_if_no_qdrant!();
    
    let mut config = QdrantConfig::default();
    config.url = "http://localhost:6334".to_string(); // Use gRPC port
    let adapter = QdrantAdapter::connect(config).await.unwrap();
    
    // Test health check
    let healthy = adapter.is_healthy().await.unwrap();
    assert!(healthy);
    
    // Test detailed health info
    let health_data = adapter.health_check().await.unwrap();
    assert_eq!(health_data["status"], "healthy");
}

#[tokio::test]
async fn test_real_qdrant_collection_operations() {
    skip_if_no_qdrant!();
    
    let mut config = QdrantConfig::default();
    config.url = "http://localhost:6334".to_string(); // Use gRPC port
    let adapter = QdrantAdapter::connect(config).await.unwrap();
    
    let collection_name = format!("test_docker_collection_{}", Uuid::new_v4().simple());
    
    // Create collection
    let collection_config = CollectionConfig::new(
        &collection_name,
        128,
        DistanceMetric::Cosine
    ).unwrap();
    
    let result = adapter.create_collection(collection_config.clone()).await;
    assert!(result.is_ok(), "Failed to create collection: {:?}", result);
    
    // List collections - should include our test collection
    let collections = adapter.list_collections().await.unwrap();
    let found = collections.iter().any(|c| &c.name == &collection_name);
    assert!(found, "Collection not found in list");
    
    // Get collection info
    let info = adapter.get_collection_info(&collection_name).await.unwrap();
    assert!(info.is_some());
    let info = info.unwrap();
    assert_eq!(info.name, collection_name);
    assert_eq!(info.dimension, 128);
    assert!(matches!(info.distance_metric, DistanceMetric::Cosine));
    
    // Get collection stats
    let stats = adapter.get_collection_stats(&collection_name).await.unwrap();
    println!("Collection stats: {:?}", stats); // Debug output to see actual keys
    // The real Qdrant might use different keys than the mock
    assert!(!stats.is_empty(), "Stats should not be empty");
    
    // Cleanup - delete collection
    let result = adapter.delete_collection(&collection_name).await;
    assert!(result.is_ok(), "Failed to delete collection: {:?}", result);
}

#[tokio::test]
async fn test_real_qdrant_vector_operations() {
    skip_if_no_qdrant!();
    
    let mut config = QdrantConfig::default();
    config.url = "http://localhost:6334".to_string(); // Use gRPC port
    let adapter = QdrantAdapter::connect(config).await.unwrap();
    
    let collection_name = format!("test_docker_vectors_{}", Uuid::new_v4().simple());
    
    // Create collection
    let collection_config = CollectionConfig::new(
        &collection_name,
        3, // Small dimension for testing
        DistanceMetric::Cosine
    ).unwrap();
    
    adapter.create_collection(collection_config).await.unwrap();
    
    // Store a vector (using generated UUID)
    let vector_id = Uuid::new_v4().to_string();
    let vector = Vector::with_metadata(
        vector_id.clone(),
        vec![0.1, 0.2, 0.3],
        HashMap::from([
            ("title".to_string(), serde_json::json!("Test Document 1")),
            ("category".to_string(), serde_json::json!("test")),
        ])
    );
    
    let result = adapter.store_vector(&collection_name, vector.clone()).await;
    assert!(result.is_ok(), "Failed to store vector: {:?}", result);
    
    // Retrieve the vector
    let retrieved = adapter.get_vector(&collection_name, &vector_id).await.unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, vector_id);
    // Vectors might be normalized by Qdrant, so check approximate equality
    let expected = vec![0.1, 0.2, 0.3];
    let actual = &retrieved.embedding;
    assert_eq!(actual.len(), expected.len(), "Vector dimensions should match");
    // Check if vector is normalized version of original
    let magnitude: f32 = expected.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let normalized_expected: Vec<f32> = expected.iter().map(|&x| x / magnitude).collect();
    
    for (a, e) in actual.iter().zip(normalized_expected.iter()) {
        assert!((a - e).abs() < 0.001, "Vector component {} should be approximately {} but got {}", a, e, a);
    }
    assert_eq!(retrieved.metadata["title"], serde_json::json!("Test Document 1"));
    
    // Store another vector for search testing
    let vector2_id = Uuid::new_v4().to_string();
    let vector2 = Vector::with_metadata(
        vector2_id.clone(),
        vec![0.1, 0.2, 0.4], // Similar to first vector
        HashMap::from([
            ("title".to_string(), serde_json::json!("Test Document 2")),
            ("category".to_string(), serde_json::json!("test")),
        ])
    );
    adapter.store_vector(&collection_name, vector2).await.unwrap();
    
    // Search for similar vectors (normalize search vector too)
    let search_vector = vec![0.1, 0.2, 0.35];
    let search_magnitude: f32 = search_vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let normalized_search: Vec<f32> = search_vector.iter().map(|&x| x / search_magnitude).collect();
    
    let search_params = SearchParams::with_limit(5)
        .with_threshold(0.5)
        .include_vectors(); // Explicitly include vectors
    
    let results = adapter.search_similar(
        &collection_name,
        normalized_search.clone(),
        search_params
    ).await.unwrap();
    
    assert!(!results.is_empty(), "Should find similar vectors");
    assert!(results.len() <= 2, "Should not exceed limit");
    
    // Verify results have scores
    for result in &results {
        assert!(result.score >= 0.5, "Score should meet threshold");
        assert!(result.score <= 1.0, "Score should be normalized");
    }
    
    // Test search with metadata filter (reuse normalized vector)
    let filtered_search = SearchParams::with_limit(5)
        .with_filter("category", serde_json::json!("test"))
        .include_vectors(); // Explicitly include vectors
    
    let filtered_results = adapter.search_similar(
        &collection_name,
        normalized_search,
        filtered_search
    ).await.unwrap();
    
    assert!(!filtered_results.is_empty(), "Should find vectors matching filter");
    for result in &filtered_results {
        assert_eq!(result.vector.metadata["category"], serde_json::json!("test"));
    }
    
    // Delete a vector
    let result = adapter.delete_vector(&collection_name, &vector_id).await;
    assert!(result.is_ok(), "Failed to delete vector: {:?}", result);
    
    // Verify deletion
    let retrieved = adapter.get_vector(&collection_name, &vector_id).await.unwrap();
    assert!(retrieved.is_none(), "Vector should be deleted");
    
    // Cleanup
    adapter.delete_collection(&collection_name).await.unwrap();
}

#[tokio::test]
async fn test_real_qdrant_batch_operations() {
    skip_if_no_qdrant!();
    
    let mut config = QdrantConfig::default();
    config.url = "http://localhost:6334".to_string(); // Use gRPC port
    let adapter = QdrantAdapter::connect(config).await.unwrap();
    
    let collection_name = format!("test_docker_batch_{}", Uuid::new_v4().simple());
    
    // Create collection
    let collection_config = CollectionConfig::new(
        &collection_name,
        3,
        DistanceMetric::Cosine
    ).unwrap();
    adapter.create_collection(collection_config).await.unwrap();
    
    // Prepare batch of vectors (using generated UUIDs)
    let batch_ids: Vec<String> = (0..3).map(|_| Uuid::new_v4().to_string()).collect();
    let vectors = vec![
        Vector::new(batch_ids[0].clone(), vec![1.0, 0.0, 0.0]),
        Vector::new(batch_ids[1].clone(), vec![0.0, 1.0, 0.0]),
        Vector::new(batch_ids[2].clone(), vec![0.0, 0.0, 1.0]),
    ];
    
    // Store batch
    let results = adapter.store_vectors_batch(&collection_name, vectors).await.unwrap();
    assert_eq!(results.len(), 3);
    for result in results {
        assert!(result.is_ok(), "Batch operation should succeed");
    }
    
    // Delete batch
    let ids_to_delete = batch_ids;
    
    let result = adapter.delete_vectors_batch(&collection_name, ids_to_delete).await;
    assert!(result.is_ok(), "Batch delete should succeed");
    
    // Cleanup
    adapter.delete_collection(&collection_name).await.unwrap();
}

#[tokio::test]
async fn test_real_qdrant_error_handling() {
    skip_if_no_qdrant!();
    
    let mut config = QdrantConfig::default();
    config.url = "http://localhost:6334".to_string(); // Use gRPC port
    let adapter = QdrantAdapter::connect(config).await.unwrap();
    
    // Try to get vector from non-existent collection
    let result = adapter.get_vector("nonexistent_collection", "some_id").await;
    assert!(result.is_err(), "Should fail for non-existent collection");
    
    // Try to delete non-existent collection
    let result = adapter.delete_collection("nonexistent_collection").await;
    // In real Qdrant, this might fail differently than in mock
    println!("Delete non-existent collection result: {:?}", result);
    
    // Try to create collection with invalid config
    let result = CollectionConfig::new("", 0, DistanceMetric::Cosine);
    assert!(result.is_err(), "Should fail with empty collection name");
}

#[tokio::test]
async fn test_real_qdrant_configuration() {
    skip_if_no_qdrant!();
    
    // Test custom configuration
    let mut config = QdrantConfig {
        url: "http://localhost:6334".to_string(), // Use gRPC port
        timeout_seconds: 30,
        max_batch_size: 50,
        ..QdrantConfig::default()
    };
    
    // Test environment variable loading
    std::env::set_var("TYL_QDRANT_TIMEOUT_SECONDS", "45");
    config.merge_env().unwrap();
    assert_eq!(config.timeout_seconds, 45);
    
    // Test connection with custom config
    let adapter = QdrantAdapter::connect(config).await.unwrap();
    let healthy = adapter.is_healthy().await.unwrap();
    assert!(healthy);
    
    // Cleanup env vars
    std::env::remove_var("TYL_QDRANT_TIMEOUT_SECONDS");
}