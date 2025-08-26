//! # Schema Migration and Advanced Filtering Example
//!
//! This example demonstrates the advanced features of the TYL Qdrant Adapter:
//! - Schema migrations with Pact.io validation
//! - Sophisticated filtering capabilities
//! - Migration management and rollback

use tyl_qdrant_adapter::{CollectionConfig, DistanceMetric, SearchParams, Vector};
use tyl_qdrant_adapter::{MockQdrantAdapter, QdrantAdapter};
use tyl_qdrant_adapter::{VectorCollectionManager, VectorStore};

#[cfg(feature = "schema-migration")]
use tyl_qdrant_adapter::migration::*;

use semver::Version;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 TYL Qdrant Adapter - Migration and Advanced Filtering Example");

    // Use MockQdrantAdapter for demonstration (no Qdrant server required)
    let adapter = MockQdrantAdapter::new();

    // Demonstrate advanced filtering capabilities
    demonstrate_advanced_filtering(&adapter).await?;

    // Demonstrate schema migration (if feature is enabled)
    #[cfg(feature = "schema-migration")]
    demonstrate_schema_migration(adapter).await?;

    #[cfg(not(feature = "schema-migration"))]
    {
        println!("\n💡 To see schema migration features, run:");
        println!("   cargo run --example migration_example --features schema-migration");
    }

    Ok(())
}

async fn demonstrate_advanced_filtering(
    adapter: &MockQdrantAdapter,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Advanced Filtering Capabilities");

    // Create a test collection
    let collection_config = CollectionConfig::new("advanced_search", 128, DistanceMetric::Cosine)?;
    adapter.create_collection(collection_config).await?;

    // Store vectors with rich metadata
    let test_vectors = vec![
        create_rich_vector("doc_1", "Technical documentation", "published", 5, true),
        create_rich_vector("doc_2", "User manual", "draft", 3, false),
        create_rich_vector("doc_3", "API reference", "published", 5, true),
        create_rich_vector("doc_4", "Tutorial guide", "review", 4, true),
        create_rich_vector("doc_5", "Release notes", "published", 2, false),
    ];

    // Store all vectors
    for vector in test_vectors {
        adapter.store_vector("advanced_search", vector).await?;
    }

    println!("✅ Stored 5 test vectors with metadata");

    // Test 1: Simple filtering
    println!("\n📋 Test 1: Simple filtering (published documents)");
    let search_params =
        SearchParams::with_limit(10).with_filter("status", serde_json::json!("published"));

    let results = adapter
        .search_similar("advanced_search", vec![0.1; 128], search_params)
        .await?;

    println!("   Found {} published documents", results.len());
    for result in &results {
        if let Some(title) = result.vector.metadata.get("title") {
            println!("   - {}", title.as_str().unwrap_or("N/A"));
        }
    }

    // Test 2: Numeric filtering
    println!("\n🔢 Test 2: Numeric filtering (high priority >= 4)");
    let search_params = SearchParams::with_limit(10)
        .with_filter("priority", serde_json::json!(5))
        .with_filter("status", serde_json::json!("published"));

    let results = adapter
        .search_similar("advanced_search", vec![0.2; 128], search_params)
        .await?;

    println!(
        "   Found {} high-priority published documents",
        results.len()
    );

    // Test 3: Boolean filtering
    println!("\n✅ Test 3: Boolean filtering (featured documents)");
    let search_params =
        SearchParams::with_limit(10).with_filter("featured", serde_json::json!(true));

    let results = adapter
        .search_similar("advanced_search", vec![0.3; 128], search_params)
        .await?;

    println!("   Found {} featured documents", results.len());

    // Test 4: Complex filtering with multiple conditions
    println!("\n🎯 Test 4: Complex filtering (published AND featured)");
    let search_params = SearchParams::with_limit(10)
        .with_filter("status", serde_json::json!("published"))
        .with_filter("featured", serde_json::json!(true))
        .with_threshold(0.1);

    let results = adapter
        .search_similar("advanced_search", vec![0.4; 128], search_params)
        .await?;

    println!(
        "   Found {} published and featured documents",
        results.len()
    );
    for result in &results {
        let title = result
            .vector
            .metadata
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("N/A");
        let priority = result
            .vector
            .metadata
            .get("priority")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        println!(
            "   - {} (priority: {}, score: {:.3})",
            title, priority, result.score
        );
    }

    // Test 5: Range filtering demonstration (using public methods)
    println!("\n📊 Test 5: Range filtering capabilities");
    if let Some(_filter) = QdrantAdapter::build_range_filter(
        "created_timestamp",
        Some(1640995200.0),
        Some(1672531200.0),
    ) {
        println!("   ✅ Range filter created for timestamp range");
        println!(
            "   📝 This filter would find documents created between Jan 1, 2022 and Jan 1, 2023"
        );
    }

    // Test 6: Complex logical filtering
    println!("\n🧠 Test 6: Complex logical filtering");
    let must_conditions = vec![
        ("category".to_string(), serde_json::json!("documentation")),
        ("status".to_string(), serde_json::json!("published")),
    ];

    let should_conditions = vec![
        ("priority".to_string(), serde_json::json!(5)),
        ("featured".to_string(), serde_json::json!(true)),
    ];

    if let Some(_filter) = QdrantAdapter::build_complex_filter(
        must_conditions,
        should_conditions,
        vec![], // no must_not conditions
    ) {
        println!("   ✅ Complex filter created with MUST and SHOULD conditions");
        println!("   📝 This filter requires: documentation + published");
        println!("   📝 And prefers: high priority OR featured");
    }

    println!("\n🎉 Advanced filtering demonstration completed!");

    Ok(())
}

#[cfg(feature = "schema-migration")]
async fn demonstrate_schema_migration(
    adapter: MockQdrantAdapter,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📋 Schema Migration with Pact.io Validation");

    let manager = SchemaMigrationManager::new(adapter);

    // Initialize migration system
    println!("🏗️  Initializing migration system...");
    manager.initialize().await?;
    println!("   ✅ Migration system initialized");

    // Create first migration
    println!("\n📦 Creating initial schema migration (v1.0.0)...");
    let migration_v1 = MigrationBuilder::new(Version::new(1, 0, 0), "Initial schema".to_string())
        .author("DevOps Team".to_string())
        .description("Create initial collections for document management".to_string())
        .create_collection(CollectionConfig::new(
            "documents",
            768,
            DistanceMetric::Cosine,
        )?)
        .create_collection(CollectionConfig::new(
            "users",
            256,
            DistanceMetric::Euclidean,
        )?)
        .add_pact_contract(create_sample_pact_contract("document-service"))
        .build();

    // Apply migration
    let result = manager.apply_migration(migration_v1).await?;
    println!("   ✅ Migration {} applied successfully", result.version);
    println!("   📝 Applied {} changes", result.applied_changes.len());
    println!(
        "   🔍 Pact validation: {}",
        if result.pact_validation_passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );

    // Create second migration with dependency
    println!("\n📦 Creating dependent migration (v1.1.0)...");
    let migration_v1_1 = MigrationBuilder::new(
        Version::new(1, 1, 0),
        "Add analytics collection".to_string(),
    )
    .author("Analytics Team".to_string())
    .description("Add collection for user analytics and behavior tracking".to_string())
    .depends_on(Version::new(1, 0, 0)) // Depends on v1.0.0
    .create_collection(CollectionConfig::new(
        "analytics",
        512,
        DistanceMetric::DotProduct,
    )?)
    .add_pact_contract(create_sample_pact_contract("analytics-service"))
    .build();

    // Apply dependent migration
    let result = manager.apply_migration(migration_v1_1).await?;
    println!("   ✅ Migration {} applied successfully", result.version);

    // Create breaking change migration
    println!("\n📦 Creating breaking change migration (v2.0.0)...");
    let migration_v2 =
        MigrationBuilder::new(Version::new(2, 0, 0), "Major schema update".to_string())
            .author("Platform Team".to_string())
            .description("Add multi-modal support with image and audio collections".to_string())
            .depends_on(Version::new(1, 1, 0))
            .breaking_change() // Mark as breaking
            .create_collection(CollectionConfig::new(
                "images",
                2048,
                DistanceMetric::Cosine,
            )?)
            .create_collection(CollectionConfig::new(
                "audio",
                1024,
                DistanceMetric::Manhattan,
            )?)
            .build();

    // Apply breaking change migration
    let is_breaking_change = migration_v2.metadata.breaking_change;
    let result = manager.apply_migration(migration_v2).await?;
    println!("   ✅ Migration {} applied successfully", result.version);
    if is_breaking_change {
        println!("   ⚠️  This is a BREAKING CHANGE - review consumer compatibility");
    }

    // Display migration history
    println!("\n📚 Migration History:");
    let history = manager.get_migration_history().await?;
    for (i, migration) in history.iter().enumerate() {
        let breaking_indicator = if migration.metadata.breaking_change {
            "⚠️ "
        } else {
            "✅ "
        };
        println!(
            "   {}{}. {} - {} (by {})",
            breaking_indicator,
            i + 1,
            migration.version,
            migration.name,
            migration.metadata.author
        );

        if !migration.metadata.dependencies.is_empty() {
            println!("      Dependencies: {:?}", migration.metadata.dependencies);
        }

        println!(
            "      Collections: {} changes",
            migration.collection_changes.len()
        );
        println!("      Pact contracts: {}", migration.pact_contracts.len());
    }

    // Demonstrate rollback
    println!("\n↩️  Demonstrating rollback...");
    if let Some(latest) = history.last() {
        if latest.metadata.reversible {
            println!("   Rolling back migration {}...", latest.version);
            manager.rollback_migration(latest.version.clone()).await?;
            println!("   ✅ Rollback completed successfully");
        } else {
            println!("   ⚠️  Latest migration is not reversible");
        }
    }

    // Show updated history after rollback
    println!("\n📚 Updated Migration History (after rollback):");
    let updated_history = manager.get_migration_history().await?;
    println!("   Current migrations: {}", updated_history.len());
    if let Some(current) = updated_history.last() {
        println!("   Current schema version: {}", current.version);
    }

    println!("\n🎉 Schema migration demonstration completed!");
    println!("💡 In production, Pact contracts would be validated against real consumer services");

    Ok(())
}

fn create_rich_vector(
    id: &str,
    title: &str,
    status: &str,
    priority: i32,
    featured: bool,
) -> Vector {
    let mut metadata = HashMap::new();
    metadata.insert("title".to_string(), serde_json::json!(title));
    metadata.insert("status".to_string(), serde_json::json!(status));
    metadata.insert("priority".to_string(), serde_json::json!(priority));
    metadata.insert("featured".to_string(), serde_json::json!(featured));
    metadata.insert("category".to_string(), serde_json::json!("documentation"));
    metadata.insert(
        "created_timestamp".to_string(),
        serde_json::json!(1672531200.0),
    ); // Jan 1, 2023
    metadata.insert("author".to_string(), serde_json::json!("System"));

    // Create a meaningful embedding based on content
    let mut embedding = vec![0.0; 128];
    // Simple hash-based embedding for demo
    let hash = title.len() as f32 * 0.01;
    embedding[0] = hash;
    embedding[1] = priority as f32 * 0.1;
    embedding[2] = if featured { 1.0 } else { 0.0 };

    Vector::with_metadata(id.to_string(), embedding, metadata)
}

#[cfg(feature = "schema-migration")]
fn create_sample_pact_contract(consumer_service: &str) -> PactContract {
    PactContract {
        consumer: consumer_service.to_string(),
        provider: "qdrant-adapter".to_string(),
        contract_path: format!("./pacts/{}-qdrant-adapter.json", consumer_service),
        interactions: vec![
            PactInteraction {
                description: format!("create collection for {}", consumer_service),
                request: VectorRequest {
                    operation: VectorOperation::CreateCollection,
                    collection: consumer_service.replace('-', "_"),
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
            },
            PactInteraction {
                description: format!("store vector in {} collection", consumer_service),
                request: VectorRequest {
                    operation: VectorOperation::StoreVector,
                    collection: consumer_service.replace('-', "_"),
                    parameters: serde_json::json!({
                        "id": "test-vector",
                        "embedding": [0.1, 0.2, 0.3]
                    }),
                },
                response: VectorResponse {
                    status: ResponseStatus::Success,
                    data: Some(serde_json::json!({"stored": true})),
                    error: None,
                },
            },
            PactInteraction {
                description: format!("search similar vectors in {} collection", consumer_service),
                request: VectorRequest {
                    operation: VectorOperation::SearchSimilar,
                    collection: consumer_service.replace('-', "_"),
                    parameters: serde_json::json!({
                        "query": [0.1, 0.2, 0.3],
                        "limit": 5,
                        "threshold": 0.8
                    }),
                },
                response: VectorResponse {
                    status: ResponseStatus::Success,
                    data: Some(serde_json::json!({
                        "results": [
                            {"id": "test-vector", "score": 0.95}
                        ]
                    })),
                    error: None,
                },
            },
        ],
    }
}
