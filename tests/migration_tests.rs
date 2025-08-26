//! Tests for schema migration functionality with Pact.io validation
//!
//! These tests verify the schema migration system including:
//! - Migration creation and application
//! - Pact contract validation
//! - Rollback functionality
//! - Dependency tracking

#[cfg(feature = "schema-migration")]
mod migration_tests {
    use tyl_qdrant_adapter::{
        MockQdrantAdapter, CollectionConfig, DistanceMetric,
        migration::*,
    };
    use semver::Version;

    #[tokio::test]
    async fn test_migration_manager_initialization() {
        let adapter = MockQdrantAdapter::new();
        let manager = SchemaMigrationManager::new(adapter);
        
        let result = manager.initialize().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_migration_builder() {
        let version = Version::new(1, 0, 0);
        let migration = MigrationBuilder::new(version.clone(), "Test migration".to_string())
            .author("Test Author".to_string())
            .description("Test description".to_string())
            .create_collection(
                CollectionConfig::new("test_collection", 128, DistanceMetric::Cosine).unwrap()
            )
            .build();

        assert_eq!(migration.version, version);
        assert_eq!(migration.name, "Test migration");
        assert_eq!(migration.metadata.author, "Test Author");
        assert_eq!(migration.collection_changes.len(), 1);
        
        match &migration.collection_changes[0] {
            CollectionChange::CreateCollection(config) => {
                assert_eq!(config.name, "test_collection");
                assert_eq!(config.dimension, 128);
            },
            _ => panic!("Expected CreateCollection change"),
        }
    }

    #[tokio::test]
    async fn test_migration_application() {
        let adapter = MockQdrantAdapter::new();
        let manager = SchemaMigrationManager::new(adapter);
        
        // Initialize migration system
        manager.initialize().await.unwrap();
        
        // Create test migration
        let version = Version::new(1, 0, 0);
        let migration = MigrationBuilder::new(version.clone(), "Create test collection".to_string())
            .author("Test".to_string())
            .create_collection(
                CollectionConfig::new("test_docs", 256, DistanceMetric::Cosine).unwrap()
            )
            .build();

        // Apply migration
        let result = manager.apply_migration(migration).await;
        assert!(result.is_ok());
        
        let migration_result = result.unwrap();
        assert_eq!(migration_result.version, version);
        assert_eq!(migration_result.applied_changes.len(), 1);
        assert!(migration_result.pact_validation_passed);
    }

    #[tokio::test]
    async fn test_migration_with_pact_contract() {
        let adapter = MockQdrantAdapter::new();
        let manager = SchemaMigrationManager::new(adapter);
        
        manager.initialize().await.unwrap();
        
        // Create migration with Pact contract
        let pact_contract = PactContract {
            consumer: "test-service".to_string(),
            provider: "qdrant-adapter".to_string(),
            contract_path: "./test-pact.json".to_string(),
            interactions: vec![
                PactInteraction {
                    description: "create collection for documents".to_string(),
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

        let version = Version::new(1, 1, 0);
        let migration = MigrationBuilder::new(version.clone(), "Add documents collection".to_string())
            .author("Test".to_string())
            .create_collection(
                CollectionConfig::new("documents", 768, DistanceMetric::Cosine).unwrap()
            )
            .add_pact_contract(pact_contract)
            .build();

        // Apply migration with Pact validation
        let result = manager.apply_migration(migration).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_migration_dependencies() {
        let adapter = MockQdrantAdapter::new();
        let manager = SchemaMigrationManager::new(adapter);
        
        manager.initialize().await.unwrap();
        
        // Apply first migration
        let v1 = Version::new(1, 0, 0);
        let migration1 = MigrationBuilder::new(v1.clone(), "Base migration".to_string())
            .author("Test".to_string())
            .create_collection(
                CollectionConfig::new("base_collection", 128, DistanceMetric::Cosine).unwrap()
            )
            .build();
        
        manager.apply_migration(migration1).await.unwrap();
        
        // Create dependent migration
        let v2 = Version::new(2, 0, 0);
        let migration2 = MigrationBuilder::new(v2.clone(), "Dependent migration".to_string())
            .author("Test".to_string())
            .depends_on(v1)
            .create_collection(
                CollectionConfig::new("dependent_collection", 256, DistanceMetric::Euclidean).unwrap()
            )
            .build();
        
        // Should succeed because dependency exists
        let result = manager.apply_migration(migration2).await;
        assert!(result.is_ok());
        
        // Test missing dependency
        let v3 = Version::new(3, 0, 0);
        let missing_dep = Version::new(99, 0, 0); // Non-existent dependency
        let migration3 = MigrationBuilder::new(v3, "Failing migration".to_string())
            .author("Test".to_string())
            .depends_on(missing_dep)
            .create_collection(
                CollectionConfig::new("failing_collection", 512, DistanceMetric::DotProduct).unwrap()
            )
            .build();
        
        // Should fail due to missing dependency
        let result = manager.apply_migration(migration3).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_migration_rollback() {
        let adapter = MockQdrantAdapter::new();
        let manager = SchemaMigrationManager::new(adapter);
        
        manager.initialize().await.unwrap();
        
        // Create reversible migration
        let version = Version::new(1, 0, 0);
        let migration = MigrationBuilder::new(version.clone(), "Reversible migration".to_string())
            .author("Test".to_string())
            .create_collection(
                CollectionConfig::new("temp_collection", 128, DistanceMetric::Cosine).unwrap()
            )
            .build(); // Default is reversible
        
        // Apply migration
        manager.apply_migration(migration).await.unwrap();
        
        // Rollback migration
        let result = manager.rollback_migration(version).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_non_reversible_migration_rollback() {
        let adapter = MockQdrantAdapter::new();
        let manager = SchemaMigrationManager::new(adapter);
        
        manager.initialize().await.unwrap();
        
        // Create non-reversible migration
        let version = Version::new(1, 0, 0);
        let migration = MigrationBuilder::new(version.clone(), "Non-reversible migration".to_string())
            .author("Test".to_string())
            .non_reversible()
            .delete_collection("old_collection".to_string())
            .build();
        
        // Apply migration
        manager.apply_migration(migration).await.unwrap();
        
        // Rollback should fail
        let result = manager.rollback_migration(version).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_migration_history() {
        let adapter = MockQdrantAdapter::new();
        let manager = SchemaMigrationManager::new(adapter);
        
        manager.initialize().await.unwrap();
        
        // Apply multiple migrations
        let migrations = vec![
            MigrationBuilder::new(Version::new(1, 0, 0), "First".to_string())
                .author("Test".to_string())
                .create_collection(
                    CollectionConfig::new("collection1", 128, DistanceMetric::Cosine).unwrap()
                )
                .build(),
            MigrationBuilder::new(Version::new(1, 1, 0), "Second".to_string())
                .author("Test".to_string())
                .create_collection(
                    CollectionConfig::new("collection2", 256, DistanceMetric::Euclidean).unwrap()
                )
                .build(),
        ];
        
        for migration in migrations {
            manager.apply_migration(migration).await.unwrap();
        }
        
        // Get migration history
        let history = manager.get_migration_history().await.unwrap();
        assert_eq!(history.len(), 2);
        
        // Should be sorted by version
        assert!(history[0].version < history[1].version);
    }

    #[test]
    fn test_pact_interaction_serialization() {
        let interaction = PactInteraction {
            description: "store vector in collection".to_string(),
            request: VectorRequest {
                operation: VectorOperation::StoreVector,
                collection: "test".to_string(),
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
        };

        // Test serialization/deserialization
        let json = serde_json::to_string(&interaction).unwrap();
        let deserialized: PactInteraction = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.description, interaction.description);
        assert!(matches!(deserialized.request.operation, VectorOperation::StoreVector));
        assert!(matches!(deserialized.response.status, ResponseStatus::Success));
    }
}

#[cfg(not(feature = "schema-migration"))]
mod disabled_tests {
    #[test]
    fn migration_feature_disabled() {
        // Test that migration module is not available without feature
        println!("Schema migration tests require 'schema-migration' feature");
    }
}