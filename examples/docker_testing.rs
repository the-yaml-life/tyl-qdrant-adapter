//! Example demonstrating Docker integration testing approach for TYL Qdrant Adapter
//!
//! This example shows how to test the adapter with a real Qdrant instance.
//! 
//! Usage:
//! 1. Start Qdrant: docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
//! 2. Run: cargo run --example docker_testing

use std::collections::HashMap;
use tyl_qdrant_adapter::{
    QdrantAdapter, QdrantConfig, ConfigPlugin,
    VectorStore, VectorCollectionManager, VectorStoreHealth, VectorDatabase,
    Vector, CollectionConfig, DistanceMetric, SearchParams,
};

// We need reqwest for the availability check
// This would normally be in Cargo.toml dev-dependencies

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 TYL Qdrant Adapter Docker Integration Example");
    
    // Check if Qdrant is available
    let qdrant_available = check_qdrant_availability().await;
    
    if !qdrant_available {
        println!("\n⚠️  Qdrant not available at localhost:6333");
        println!("   Start with: docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant");
        println!("   Then run: cargo run --example docker_testing");
        return Ok(());
    }
    
    println!("✅ Qdrant server detected at localhost:6333");
    
    // Attempt to connect with QdrantAdapter
    println!("\n🔌 Attempting to connect to Qdrant...");
    
    let config = QdrantConfig::default();
    match QdrantAdapter::connect(config).await {
        Ok(adapter) => {
            println!("✅ Successfully connected to Qdrant!");
            
            // Run basic integration test
            match run_basic_integration_test(&adapter).await {
                Ok(_) => println!("✅ Integration test passed!"),
                Err(e) => println!("❌ Integration test failed: {}", e),
            }
        }
        Err(e) => {
            println!("❌ Failed to connect to Qdrant: {}", e);
            println!("\n💡 This is likely due to version compatibility issues between");
            println!("   qdrant-client and the Qdrant server. For reliable testing, use:");
            println!("   cargo test --test integration_tests (uses MockQdrantAdapter)");
        }
    }
    
    Ok(())
}

async fn check_qdrant_availability() -> bool {
    match reqwest::get("http://localhost:6333/").await {
        Ok(response) => {
            if response.status().is_success() {
                if let Ok(body) = response.text().await {
                    println!("📋 Qdrant server info: {}", body.chars().take(100).collect::<String>());
                    return true;
                }
            }
            false
        }
        Err(_) => false,
    }
}

async fn run_basic_integration_test(adapter: &QdrantAdapter) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Running basic integration test...");
    
    // Test health check
    let healthy = adapter.is_healthy().await?;
    println!("   Health check: {}", if healthy { "✅ Healthy" } else { "❌ Unhealthy" });
    
    // Test collection creation
    let collection_name = "docker_test_collection";
    let collection_config = CollectionConfig::new(
        collection_name,
        3, // Small dimension for testing
        DistanceMetric::Cosine
    )?;
    
    adapter.create_collection(collection_config).await?;
    println!("   Collection creation: ✅ Success");
    
    // Test vector storage
    let vector = Vector::with_metadata(
        "test_vector_1".to_string(),
        vec![0.1, 0.2, 0.3],
        HashMap::from([
            ("title".to_string(), serde_json::json!("Test Document")),
            ("type".to_string(), serde_json::json!("example")),
        ])
    );
    
    adapter.store_vector(collection_name, vector).await?;
    println!("   Vector storage: ✅ Success");
    
    // Test vector retrieval
    let retrieved = adapter.get_vector(collection_name, "test_vector_1").await?;
    match retrieved {
        Some(vector) => {
            println!("   Vector retrieval: ✅ Success (ID: {})", vector.id);
        }
        None => {
            return Err("Vector not found after storage".into());
        }
    }
    
    // Test search
    let search_params = SearchParams::with_limit(5);
    let results = adapter.search_similar(
        collection_name,
        vec![0.1, 0.2, 0.35], // Similar to stored vector
        search_params
    ).await?;
    
    println!("   Similarity search: ✅ Success ({} results)", results.len());
    
    // Cleanup
    adapter.delete_collection(collection_name).await?;
    println!("   Cleanup: ✅ Success");
    
    Ok(())
}