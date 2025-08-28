//! # Advanced Filtering Example - TYL Qdrant Adapter
//! 
//! This example demonstrates the enhanced filtering capabilities implemented 
//! in the TYL Qdrant Adapter, including:
//! 
//! - Range queries using `$gte`, `$lte`, `$gt`, `$lt` operators
//! - IN queries using `$in` operator  
//! - EXISTS queries using `$exists` operator
//! - Backward compatibility with simple equality filters
//!
//! ## Usage
//! ```bash
//! cargo run --example advanced_filtering --features mock
//! ```

use tyl_qdrant_adapter::{
    CollectionConfig, DistanceMetric, MockQdrantAdapter, SearchParams, Vector, VectorCollectionManager,
    VectorStore,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 TYL Qdrant Adapter - Advanced Filtering Demo");
    println!("================================================\n");

    // Create mock adapter for demonstration
    let adapter = MockQdrantAdapter::new();

    // Create a collection for product search
    let collection_config = CollectionConfig::new("products", 128, DistanceMetric::Cosine)?;
    adapter.create_collection(collection_config).await?;

    println!("📦 Setting up sample product data...");
    
    // Store sample products with rich metadata
    let products = vec![
        ("laptop_gaming", [0.9, 0.1].iter().cycle().take(128).cloned().collect::<Vec<f32>>(), "Gaming Laptop", 1299.99, 4.5, "electronics", true),
        ("laptop_business", [0.8, 0.2].iter().cycle().take(128).cloned().collect::<Vec<f32>>(), "Business Laptop", 899.99, 4.2, "electronics", true),
        ("phone_premium", [0.7, 0.3].iter().cycle().take(128).cloned().collect::<Vec<f32>>(), "Premium Phone", 999.99, 4.8, "electronics", true),
        ("phone_budget", [0.6, 0.4].iter().cycle().take(128).cloned().collect::<Vec<f32>>(), "Budget Phone", 299.99, 3.9, "electronics", false),
        ("book_fiction", [0.2, 0.8].iter().cycle().take(128).cloned().collect::<Vec<f32>>(), "Fiction Novel", 19.99, 4.3, "books", false),
        ("book_tech", [0.3, 0.7].iter().cycle().take(128).cloned().collect::<Vec<f32>>(), "Tech Manual", 49.99, 4.1, "books", false),
        ("headphones", [0.5, 0.5].iter().cycle().take(128).cloned().collect::<Vec<f32>>(), "Wireless Headphones", 199.99, 4.4, "electronics", true),
    ];

    for (id, embedding, title, price, rating, category, premium) in products {
        let mut vector = Vector::new(id.to_string(), embedding);
        vector.add_metadata("title", serde_json::json!(title));
        vector.add_metadata("price", serde_json::json!(price));
        vector.add_metadata("rating", serde_json::json!(rating));
        vector.add_metadata("category", serde_json::json!(category));
        vector.add_metadata("premium", serde_json::json!(premium));
        
        adapter.store_vector("products", vector).await?;
    }

    println!("✅ Stored {} products with rich metadata\n", 7);

    // Demo 1: Range Filtering
    println!("🔍 Demo 1: Range Filtering");
    println!("==========================");
    println!("Searching for products priced between $200 and $1000...");
    
    let range_filter = serde_json::json!({
        "$gte": 200.0,
        "$lte": 1000.0
    });
    
    let search_params = SearchParams::with_limit(5)
        .with_filter("price", range_filter)
        .include_vectors();

    // Use a query vector that might match electronics
    let query_vector: Vec<f32> = [0.8, 0.2].iter().cycle().take(128).cloned().collect();
    
    match adapter.search_similar("products", query_vector, search_params).await {
        Ok(results) => {
            println!("📋 Found {} products in price range $200-$1000:", results.len());
            for result in results {
                println!("   • {} (score: {:.3})", result.vector.id, result.score);
                if let Some(title) = result.vector.metadata.get("title") {
                    if let Some(price) = result.vector.metadata.get("price") {
                        println!("     Title: {}, Price: ${}", title.as_str().unwrap_or("N/A"), price.as_f64().unwrap_or(0.0));
                    }
                }
            }
        },
        Err(e) => println!("❌ Range filtering: {}", e),
    }
    println!();

    // Demo 2: IN Filtering  
    println!("🔍 Demo 2: IN Filtering");
    println!("=======================");
    println!("Searching for electronics and books only...");
    
    let in_filter = serde_json::json!({
        "$in": ["electronics", "books"]
    });
    
    let search_params = SearchParams::with_limit(5)
        .with_filter("category", in_filter);

    let query_vector: Vec<f32> = [0.5, 0.5].iter().cycle().take(128).cloned().collect();
    
    match adapter.search_similar("products", query_vector, search_params).await {
        Ok(results) => {
            println!("📋 Found {} products in categories [electronics, books]:", results.len());
            for result in results {
                println!("   • {} (score: {:.3})", result.vector.id, result.score);
                if let Some(category) = result.vector.metadata.get("category") {
                    println!("     Category: {}", category.as_str().unwrap_or("N/A"));
                }
            }
        },
        Err(e) => println!("❌ IN filtering: {}", e),
    }
    println!();

    // Demo 3: EXISTS Filtering
    println!("🔍 Demo 3: EXISTS Filtering");
    println!("===========================");
    println!("Searching for products that have premium status...");
    
    let exists_filter = serde_json::json!({
        "$exists": true
    });
    
    let search_params = SearchParams::with_limit(5)
        .with_filter("premium", exists_filter);

    let query_vector: Vec<f32> = [0.6, 0.4].iter().cycle().take(128).cloned().collect();
    
    match adapter.search_similar("products", query_vector, search_params).await {
        Ok(results) => {
            println!("📋 Found {} products with premium field:", results.len());
            for result in results {
                println!("   • {} (score: {:.3})", result.vector.id, result.score);
                if let Some(premium) = result.vector.metadata.get("premium") {
                    println!("     Premium: {}", premium.as_bool().unwrap_or(false));
                }
            }
        },
        Err(e) => println!("❌ EXISTS filtering: {}", e),
    }
    println!();

    // Demo 4: Combined Filtering (traditional + advanced)
    println!("🔍 Demo 4: Combined Filtering");
    println!("=============================");
    println!("Searching for premium electronics under $1200...");
    
    let range_filter = serde_json::json!({
        "$lt": 1200.0
    });
    
    let search_params = SearchParams::with_limit(5)
        .with_filter("category", serde_json::json!("electronics"))  // Traditional
        .with_filter("premium", serde_json::json!(true))            // Traditional  
        .with_filter("price", range_filter);                        // Advanced

    let query_vector: Vec<f32> = [0.7, 0.3].iter().cycle().take(128).cloned().collect();
    
    match adapter.search_similar("products", query_vector, search_params).await {
        Ok(results) => {
            println!("📋 Found {} premium electronics under $1200:", results.len());
            for result in results {
                println!("   • {} (score: {:.3})", result.vector.id, result.score);
                if let Some(title) = result.vector.metadata.get("title") {
                    if let Some(price) = result.vector.metadata.get("price") {
                        println!("     {}: ${:.2}", title.as_str().unwrap_or("N/A"), price.as_f64().unwrap_or(0.0));
                    }
                }
            }
        },
        Err(e) => println!("❌ Combined filtering: {}", e),
    }
    println!();

    // Demo 5: Backward Compatibility
    println!("🔍 Demo 5: Backward Compatibility");
    println!("==================================");
    println!("Traditional simple filters still work perfectly...");
    
    let search_params = SearchParams::with_limit(5)
        .with_filter("category", serde_json::json!("books"));

    let query_vector: Vec<f32> = [0.25, 0.75].iter().cycle().take(128).cloned().collect();
    
    match adapter.search_similar("products", query_vector, search_params).await {
        Ok(results) => {
            println!("📋 Found {} books using traditional filtering:", results.len());
            for result in results {
                println!("   • {} (score: {:.3})", result.vector.id, result.score);
                if let Some(title) = result.vector.metadata.get("title") {
                    println!("     Title: {}", title.as_str().unwrap_or("N/A"));
                }
            }
        },
        Err(e) => println!("❌ Traditional filtering: {}", e),
    }
    println!();

    println!("📊 Filtering Syntax Reference");
    println!("=============================");
    println!("Range Filters:");
    println!("  • {{\"$gte\": 100.0}} - Greater than or equal");
    println!("  • {{\"$lte\": 500.0}} - Less than or equal"); 
    println!("  • {{\"$gt\": 50.0}}   - Greater than");
    println!("  • {{\"$lt\": 1000.0}} - Less than");
    println!("  • {{\"$gte\": 100.0, \"$lte\": 500.0}} - Range");
    println!();
    println!("Array Filters:");
    println!("  • {{\"$in\": [\"electronics\", \"books\"]}} - Value in list");
    println!();
    println!("Existence Filters:");
    println!("  • {{\"$exists\": true}}  - Field exists");
    println!("  • {{\"$exists\": false}} - Field missing");
    println!();
    println!("Traditional (still supported):");
    println!("  • \"exact_value\"     - Exact string match");
    println!("  • 42                - Exact number match");
    println!("  • true              - Exact boolean match");
    println!();

    println!("✅ Advanced filtering demo completed!");
    println!("🔧 Note: This demo uses MockQdrantAdapter for demonstration.");
    println!("🚀 With real Qdrant, these filters provide efficient server-side filtering.");

    Ok(())
}