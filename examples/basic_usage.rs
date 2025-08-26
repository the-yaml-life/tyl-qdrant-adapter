//! # Basic Usage Example for TYL Qdrant Adapter
//!
//! This example demonstrates how to use the Qdrant adapter with the TYL framework.
//! It shows basic vector operations including collection creation, vector storage, and similarity search.

use tyl_qdrant_adapter::{QdrantAdapter, QdrantConfig, ConfigPlugin};
use tyl_vector_port::{
    CollectionConfig, DistanceMetric, SearchParams, Vector, VectorDatabase,
    VectorStore, VectorCollectionManager, VectorStoreHealth
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 TYL Qdrant Adapter - Basic Usage Example");
    
    // Create Qdrant configuration using TYL config patterns
    // This will automatically load from environment variables if set:
    // - TYL_QDRANT_URL or QDRANT_URL
    // - TYL_QDRANT_API_KEY or QDRANT_API_KEY
    // - etc.
    let mut config = QdrantConfig::default();
    config.merge_env()?;
    
    println!("📝 Configuration:");
    println!("   URL: {}", config.url);
    println!("   Timeout: {}s", config.timeout_seconds);
    println!("   Max Batch Size: {}", config.max_batch_size);
    println!("   Compression: {}", config.enable_compression);
    
    // Connect to Qdrant using the VectorDatabase trait
    println!("\n🔌 Connecting to Qdrant...");
    let adapter = match QdrantAdapter::connect(config).await {
        Ok(adapter) => {
            println!("✅ Connected successfully!");
            println!("   {}", adapter.connection_info());
            adapter
        }
        Err(e) => {
            eprintln!("❌ Failed to connect to Qdrant: {}", e);
            eprintln!("💡 Make sure Qdrant is running at http://localhost:6333");
            eprintln!("   You can start Qdrant with: docker run -p 6333:6333 qdrant/qdrant");
            return Ok(());
        }
    };

    // Check health
    println!("\n🏥 Health Check...");
    match adapter.is_healthy().await {
        Ok(true) => println!("✅ Qdrant is healthy"),
        Ok(false) => println!("⚠️  Qdrant is unhealthy"),
        Err(e) => {
            eprintln!("❌ Health check failed: {}", e);
            return Ok(());
        }
    }

    // Create a collection for document embeddings
    let collection_name = "documents";
    println!("\n📚 Creating collection '{}'...", collection_name);
    
    let collection_config = CollectionConfig::new(
        collection_name,
        768, // Typical embedding dimension (e.g., OpenAI text-embedding-ada-002)
        DistanceMetric::Cosine
    )?;
    
    match adapter.create_collection(collection_config).await {
        Ok(_) => println!("✅ Collection created successfully"),
        Err(e) if e.to_string().contains("already exists") => {
            println!("ℹ️  Collection already exists, continuing...");
        }
        Err(e) => {
            eprintln!("❌ Failed to create collection: {}", e);
            return Ok(());
        }
    }

    // Create some sample vectors (simulating document embeddings)
    println!("\n📄 Creating sample document vectors...");
    let vectors = create_sample_vectors();
    
    // Store vectors individually
    println!("💾 Storing vectors...");
    for (i, vector) in vectors.iter().enumerate() {
        match adapter.store_vector(collection_name, vector.clone()).await {
            Ok(_) => println!("   ✅ Stored vector {}: '{}'", i + 1, vector.id),
            Err(e) => eprintln!("   ❌ Failed to store vector {}: {}", i + 1, e),
        }
    }

    // Wait a moment for indexing (Qdrant needs time to index vectors)
    println!("\n⏳ Waiting for indexing...");
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Perform similarity search
    println!("\n🔍 Performing similarity search...");
    let query_vector = create_query_vector();
    
    let search_params = SearchParams::with_limit(3)
        .with_threshold(0.1)
        .with_filter("category", serde_json::json!("document"));
    
    match adapter.search_similar(collection_name, query_vector, search_params).await {
        Ok(results) => {
            println!("✅ Found {} similar documents:", results.len());
            for (i, result) in results.iter().enumerate() {
                println!("   {}. {} (similarity: {:.3})", 
                    i + 1, 
                    result.vector.id, 
                    result.score
                );
                
                // Show metadata if available
                if let Some(title) = result.vector.metadata.get("title") {
                    println!("      Title: {}", title.as_str().unwrap_or("N/A"));
                }
            }
        }
        Err(e) => eprintln!("❌ Search failed: {}", e),
    }

    // Get collection statistics
    println!("\n📊 Collection Statistics:");
    match adapter.get_collection_stats(collection_name).await {
        Ok(stats) => {
            for (key, value) in stats {
                println!("   {}: {}", key, value);
            }
        }
        Err(e) => eprintln!("❌ Failed to get stats: {}", e),
    }

    // List all collections
    println!("\n📋 Available Collections:");
    match adapter.list_collections().await {
        Ok(collections) => {
            for collection in collections {
                println!("   - {} ({} dimensions, {:?})", 
                    collection.name, 
                    collection.dimension, 
                    collection.distance_metric
                );
            }
        }
        Err(e) => eprintln!("❌ Failed to list collections: {}", e),
    }

    println!("\n✨ Example completed successfully!");
    println!("💡 Try setting environment variables like TYL_QDRANT_URL to customize configuration");
    
    Ok(())
}

/// Create sample document vectors for demonstration
fn create_sample_vectors() -> Vec<Vector> {
    vec![
        create_document_vector(
            "doc_1",
            "Introduction to Machine Learning",
            "This document covers the basics of machine learning algorithms and techniques.",
            vec![0.1, 0.2, 0.3] // Simulated embedding
        ),
        create_document_vector(
            "doc_2", 
            "Deep Learning Fundamentals",
            "A comprehensive guide to neural networks and deep learning concepts.",
            vec![0.15, 0.25, 0.28] // Similar to doc_1
        ),
        create_document_vector(
            "doc_3",
            "Rust Programming Guide", 
            "Learn systems programming with Rust language features and patterns.",
            vec![0.8, 0.1, 0.05] // Different topic
        ),
        create_document_vector(
            "doc_4",
            "Vector Databases Explained",
            "Understanding vector databases and their applications in AI.",
            vec![0.12, 0.18, 0.32] // Related to ML
        ),
    ]
}

/// Helper function to create a document vector with metadata
fn create_document_vector(id: &str, title: &str, content: &str, embedding: Vec<f32>) -> Vector {
    // Pad embedding to 768 dimensions (typical for embeddings)
    let mut padded_embedding = embedding;
    padded_embedding.resize(768, 0.0);
    
    let mut metadata = HashMap::new();
    metadata.insert("title".to_string(), serde_json::json!(title));
    metadata.insert("content".to_string(), serde_json::json!(content));
    metadata.insert("category".to_string(), serde_json::json!("document"));
    metadata.insert("created_at".to_string(), serde_json::json!(chrono::Utc::now().to_rfc3339()));
    
    Vector::with_metadata(id.to_string(), padded_embedding, metadata)
}

/// Create a query vector for similarity search
fn create_query_vector() -> Vec<f32> {
    // Simulated query embedding for "machine learning tutorial"
    let mut query = vec![0.11, 0.19, 0.31]; // Similar to ML documents
    query.resize(768, 0.0);
    query
}