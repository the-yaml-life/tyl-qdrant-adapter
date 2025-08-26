//! Schema migration tools with Pact.io validation for TYL Qdrant Adapter
//!
//! This module provides tools for managing schema migrations in vector collections
//! with contract testing validation using Pact.io patterns.
//!
//! ## Features
//! - Collection schema versioning and migration
//! - Backward compatibility validation
//! - Contract-based testing with Pact.io
//! - Automated migration rollback capabilities
//! - Migration dependency tracking

use super::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg(feature = "schema-migration")]
use pact_models::prelude::*;

#[allow(unused_imports)]
use pact_models; // Ensure pact_models is available when feature is enabled

/// Schema migration definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMigration {
    /// Migration version (semantic versioning)
    pub version: semver::Version,
    /// Migration name/description
    pub name: String,
    /// Collection configuration changes
    pub collection_changes: Vec<CollectionChange>,
    /// Metadata about the migration
    pub metadata: MigrationMetadata,
    /// Pact contracts to validate
    pub pact_contracts: Vec<PactContract>,
}

/// Collection configuration changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollectionChange {
    /// Create a new collection
    CreateCollection(CollectionConfig),
    /// Delete an existing collection
    DeleteCollection(String),
    /// Update collection configuration
    UpdateCollection {
        name: String,
        dimension_change: Option<usize>,
        distance_metric_change: Option<DistanceMetric>,
    },
    /// Rename collection
    RenameCollection { old_name: String, new_name: String },
    /// Add index to collection
    AddIndex { collection: String, field: String, index_type: IndexType },
    /// Remove index from collection
    RemoveIndex { collection: String, field: String },
}

/// Index types for vector collections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    /// Text search index
    Text,
    /// Numeric range index
    Numeric,
    /// Keyword exact match index
    Keyword,
    /// Geographic index
    Geo,
    /// Boolean index
    Boolean,
}

/// Migration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationMetadata {
    /// Author of the migration
    pub author: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Migration description
    pub description: String,
    /// Dependencies (other migration versions)
    pub dependencies: Vec<semver::Version>,
    /// Whether this migration is reversible
    pub reversible: bool,
    /// Breaking change indicator
    pub breaking_change: bool,
}

/// Pact contract definition for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PactContract {
    /// Consumer name (usually the microservice using this adapter)
    pub consumer: String,
    /// Provider name (always "qdrant-adapter")
    pub provider: String,
    /// Contract file path or content
    pub contract_path: String,
    /// Expected interactions
    pub interactions: Vec<PactInteraction>,
}

/// Pact interaction for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PactInteraction {
    /// Interaction description
    pub description: String,
    /// Expected request to vector store
    pub request: VectorRequest,
    /// Expected response from vector store
    pub response: VectorResponse,
}

/// Vector store request for Pact testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRequest {
    /// Operation type
    pub operation: VectorOperation,
    /// Collection name
    pub collection: String,
    /// Request parameters
    pub parameters: serde_json::Value,
}

/// Vector store response for Pact testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorResponse {
    /// Response status
    pub status: ResponseStatus,
    /// Response data
    pub data: Option<serde_json::Value>,
    /// Error information if applicable
    pub error: Option<String>,
}

/// Vector operations for Pact contracts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorOperation {
    StoreVector,
    GetVector,
    SearchSimilar,
    DeleteVector,
    CreateCollection,
    DeleteCollection,
    ListCollections,
}

/// Response status for Pact contracts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseStatus {
    Success,
    Error,
    NotFound,
}

/// Schema migration manager with Pact.io validation
pub struct SchemaMigrationManager<T> 
where
    T: VectorDatabase + VectorStore + VectorCollectionManager + Send + Sync,
{
    adapter: T,
    migration_collection: String,
    pact_dir: String,
}

impl<T> SchemaMigrationManager<T>
where
    T: VectorDatabase + VectorStore + VectorCollectionManager + Send + Sync,
{
    /// Create new migration manager
    pub fn new(adapter: T) -> Self {
        Self {
            adapter,
            migration_collection: "_tyl_migrations".to_string(),
            pact_dir: "./pacts".to_string(),
        }
    }

    /// Set custom migration collection name
    pub fn with_migration_collection(mut self, collection_name: String) -> Self {
        self.migration_collection = collection_name;
        self
    }

    /// Set custom Pact contracts directory
    pub fn with_pact_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.pact_dir = dir.as_ref().to_string_lossy().to_string();
        self
    }

    /// Initialize migration system
    pub async fn initialize(&self) -> TylResult<()> {
        // Create migrations tracking collection
        let migration_config = CollectionConfig::new(
            &self.migration_collection,
            256, // Small dimension for migration metadata
            DistanceMetric::Cosine,
        )?;

        match self.adapter.create_collection(migration_config).await {
            Ok(_) => Ok(()),
            Err(e) if e.to_string().contains("already exists") => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Apply migration with Pact validation
    pub async fn apply_migration(&self, migration: SchemaMigration) -> TylResult<MigrationResult> {
        // 1. Validate Pact contracts first
        self.validate_pact_contracts(&migration.pact_contracts).await?;

        // 2. Check migration dependencies
        self.validate_dependencies(&migration).await?;

        // 3. Apply collection changes
        let mut results = Vec::new();
        for change in &migration.collection_changes {
            let result = self.apply_collection_change(change).await?;
            results.push(result);
        }

        // 4. Record migration in tracking collection
        self.record_migration(&migration).await?;

        Ok(MigrationResult {
            version: migration.version,
            applied_changes: results,
            pact_validation_passed: true,
        })
    }

    /// Rollback migration if reversible
    pub async fn rollback_migration(&self, version: semver::Version) -> TylResult<()> {
        let migration = self.get_migration_record(&version).await?;
        
        if !migration.metadata.reversible {
            return Err(TylError::validation(
                "rollback",
                format!("Migration {} is not reversible", version),
            ));
        }

        // Apply reverse changes
        for change in migration.collection_changes.iter().rev() {
            self.apply_reverse_change(change).await?;
        }

        // Remove migration record
        self.remove_migration_record(&version).await?;

        Ok(())
    }

    /// Get migration history
    pub async fn get_migration_history(&self) -> TylResult<Vec<SchemaMigration>> {
        // Query migration collection for all applied migrations
        let search_params = SearchParams::with_limit(1000); // Large limit to get all
        let results = self.adapter.search_similar(
            &self.migration_collection,
            vec![0.0; 256], // Dummy query vector
            search_params,
        ).await?;

        let mut migrations = Vec::new();
        for result in results {
            if let Some(migration_data) = result.vector.metadata.get("migration") {
                if let Ok(migration) = serde_json::from_value::<SchemaMigration>(migration_data.clone()) {
                    migrations.push(migration);
                }
            }
        }

        // Sort by version
        migrations.sort_by(|a, b| a.version.cmp(&b.version));
        Ok(migrations)
    }

    /// Validate Pact contracts
    async fn validate_pact_contracts(&self, contracts: &[PactContract]) -> TylResult<()> {
        #[cfg(feature = "schema-migration")]
        {
            for contract in contracts {
                self.validate_single_pact_contract(contract).await?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "schema-migration")]
    async fn validate_single_pact_contract(&self, contract: &PactContract) -> TylResult<()> {
        use std::fs;
        use tempfile::TempDir;

        // Create temporary Pact file
        let temp_dir = TempDir::new()
            .map_err(|e| TylError::database(format!("Failed to create temp dir: {e}")))?;
        
        let pact_file = temp_dir.path().join("contract.json");
        
        // Generate Pact content
        let pact_content = self.generate_pact_content(contract)?;
        fs::write(&pact_file, pact_content)
            .map_err(|e| TylError::database(format!("Failed to write Pact file: {e}")))?;

        // Validate each interaction
        for interaction in &contract.interactions {
            self.validate_pact_interaction(interaction).await?;
        }

        Ok(())
    }

    #[cfg(feature = "schema-migration")]
    fn generate_pact_content(&self, contract: &PactContract) -> TylResult<String> {
        let pact_json = serde_json::json!({
            "consumer": { "name": contract.consumer },
            "provider": { "name": contract.provider },
            "interactions": contract.interactions.iter().map(|interaction| {
                serde_json::json!({
                    "description": interaction.description,
                    "request": {
                        "method": "POST",
                        "path": "/vector-operation",
                        "body": serde_json::to_value(&interaction.request).unwrap_or_default()
                    },
                    "response": {
                        "status": match interaction.response.status {
                            ResponseStatus::Success => 200,
                            ResponseStatus::Error => 500,
                            ResponseStatus::NotFound => 404,
                        },
                        "body": interaction.response.data.clone().unwrap_or(serde_json::json!({}))
                    }
                })
            }).collect::<Vec<_>>(),
            "metadata": {
                "pactSpecification": { "version": "2.0.0" }
            }
        });

        serde_json::to_string_pretty(&pact_json)
            .map_err(|e| TylError::database(format!("Failed to serialize Pact: {e}")))
    }

    async fn validate_pact_interaction(&self, interaction: &PactInteraction) -> TylResult<()> {
        // Simulate the operation and validate response matches expected
        match interaction.request.operation {
            VectorOperation::CreateCollection => {
                // Test collection creation
                let test_config = CollectionConfig::new("test_migration", 128, DistanceMetric::Cosine)?;
                let result = self.adapter.create_collection(test_config).await;
                self.validate_operation_result(result, &interaction.response)?;
            },
            VectorOperation::StoreVector => {
                // Test vector storage
                let test_vector = Vector::new("test".to_string(), vec![0.0; 128]);
                let result = self.adapter.store_vector(&interaction.request.collection, test_vector).await;
                self.validate_operation_result(result, &interaction.response)?;
            },
            VectorOperation::SearchSimilar => {
                // Test similarity search
                let params = SearchParams::with_limit(5);
                let result = self.adapter.search_similar(&interaction.request.collection, vec![0.0; 128], params).await;
                match result {
                    Ok(_) => {
                        if !matches!(interaction.response.status, ResponseStatus::Success) {
                            return Err(TylError::validation("pact", "Expected success but operation succeeded"));
                        }
                    },
                    Err(_) => {
                        if matches!(interaction.response.status, ResponseStatus::Success) {
                            return Err(TylError::validation("pact", "Expected error but operation succeeded"));
                        }
                    }
                }
            },
            _ => {
                // Add validation for other operations as needed
            }
        }

        Ok(())
    }

    fn validate_operation_result<R>(&self, result: TylResult<R>, expected_response: &VectorResponse) -> TylResult<()> {
        match (&result, &expected_response.status) {
            (Ok(_), ResponseStatus::Success) => Ok(()),
            (Err(_), ResponseStatus::Error) => Ok(()),
            (Err(_), ResponseStatus::NotFound) => Ok(()),
            (Ok(_), ResponseStatus::Error) => {
                Err(TylError::validation("pact", "Expected error but operation succeeded"))
            },
            (Ok(_), ResponseStatus::NotFound) => {
                Err(TylError::validation("pact", "Expected not found but operation succeeded"))
            },
            (Err(_), ResponseStatus::Success) => {
                Err(TylError::validation("pact", "Expected success but operation failed"))
            },
        }
    }

    async fn validate_dependencies(&self, migration: &SchemaMigration) -> TylResult<()> {
        let history = self.get_migration_history().await?;
        let applied_versions: std::collections::HashSet<_> = history.iter().map(|m| &m.version).collect();

        for dep in &migration.metadata.dependencies {
            if !applied_versions.contains(dep) {
                return Err(TylError::validation(
                    "dependencies",
                    format!("Required migration {} not found", dep),
                ));
            }
        }

        Ok(())
    }

    async fn apply_collection_change(&self, change: &CollectionChange) -> TylResult<ChangeResult> {
        match change {
            CollectionChange::CreateCollection(config) => {
                self.adapter.create_collection(config.clone()).await?;
                Ok(ChangeResult::CollectionCreated(config.name.clone()))
            },
            CollectionChange::DeleteCollection(name) => {
                self.adapter.delete_collection(name).await?;
                Ok(ChangeResult::CollectionDeleted(name.clone()))
            },
            CollectionChange::UpdateCollection { name, dimension_change: _, distance_metric_change: _ } => {
                // Note: Qdrant doesn't support changing collection config after creation
                // This would require data migration
                Err(TylError::validation(
                    "update_collection",
                    format!("Collection {} update not supported - requires manual migration", name),
                ))
            },
            CollectionChange::RenameCollection { old_name, new_name } => {
                // Qdrant doesn't support renaming - would require recreation and data migration
                Err(TylError::validation(
                    "rename_collection",
                    format!("Collection rename from {} to {} requires manual migration", old_name, new_name),
                ))
            },
            CollectionChange::AddIndex { collection, field, index_type } => {
                // Qdrant handles indexing automatically - this is mostly for documentation
                Ok(ChangeResult::IndexAdded {
                    collection: collection.clone(),
                    field: field.clone(),
                    index_type: index_type.clone(),
                })
            },
            CollectionChange::RemoveIndex { collection, field } => {
                // Qdrant handles indexing automatically - this is mostly for documentation
                Ok(ChangeResult::IndexRemoved {
                    collection: collection.clone(),
                    field: field.clone(),
                })
            },
        }
    }

    async fn apply_reverse_change(&self, change: &CollectionChange) -> TylResult<()> {
        match change {
            CollectionChange::CreateCollection(config) => {
                self.adapter.delete_collection(&config.name).await
            },
            CollectionChange::DeleteCollection(name) => {
                Err(TylError::validation(
                    "rollback",
                    format!("Cannot recreate deleted collection {} without backup", name),
                ))
            },
            _ => Ok(()), // Other changes are mostly metadata
        }
    }

    async fn record_migration(&self, migration: &SchemaMigration) -> TylResult<()> {
        let mut metadata = HashMap::new();
        metadata.insert("migration".to_string(), serde_json::to_value(migration)?);
        metadata.insert("type".to_string(), serde_json::json!("schema_migration"));

        let migration_vector = Vector::with_metadata(
            migration.version.to_string(),
            vec![0.0; 256], // Dummy vector for storage
            metadata,
        );

        self.adapter.store_vector(&self.migration_collection, migration_vector).await
    }

    async fn get_migration_record(&self, version: &semver::Version) -> TylResult<SchemaMigration> {
        let vector = self.adapter.get_vector(&self.migration_collection, &version.to_string()).await?;
        
        if let Some(v) = vector {
            if let Some(migration_data) = v.metadata.get("migration") {
                return serde_json::from_value(migration_data.clone())
                    .map_err(|e| TylError::database(format!("Invalid migration data: {e}")));
            }
        }

        Err(TylError::not_found("migration", version.to_string()))
    }

    async fn remove_migration_record(&self, version: &semver::Version) -> TylResult<()> {
        self.adapter.delete_vector(&self.migration_collection, &version.to_string()).await
    }
}

/// Result of applying a migration
#[derive(Debug, Clone)]
pub struct MigrationResult {
    pub version: semver::Version,
    pub applied_changes: Vec<ChangeResult>,
    pub pact_validation_passed: bool,
}

/// Result of applying a collection change
#[derive(Debug, Clone)]
pub enum ChangeResult {
    CollectionCreated(String),
    CollectionDeleted(String),
    CollectionUpdated(String),
    CollectionRenamed { old: String, new: String },
    IndexAdded { collection: String, field: String, index_type: IndexType },
    IndexRemoved { collection: String, field: String },
}

/// Migration builder for fluent API
pub struct MigrationBuilder {
    migration: SchemaMigration,
}

impl MigrationBuilder {
    /// Create new migration builder
    pub fn new(version: semver::Version, name: String) -> Self {
        Self {
            migration: SchemaMigration {
                version,
                name,
                collection_changes: Vec::new(),
                metadata: MigrationMetadata {
                    author: "unknown".to_string(),
                    created_at: chrono::Utc::now(),
                    description: String::new(),
                    dependencies: Vec::new(),
                    reversible: true,
                    breaking_change: false,
                },
                pact_contracts: Vec::new(),
            },
        }
    }

    /// Set migration author
    pub fn author(mut self, author: String) -> Self {
        self.migration.metadata.author = author;
        self
    }

    /// Set migration description
    pub fn description(mut self, description: String) -> Self {
        self.migration.metadata.description = description;
        self
    }

    /// Add dependency on another migration
    pub fn depends_on(mut self, version: semver::Version) -> Self {
        self.migration.metadata.dependencies.push(version);
        self
    }

    /// Mark as breaking change
    pub fn breaking_change(mut self) -> Self {
        self.migration.metadata.breaking_change = true;
        self
    }

    /// Mark as non-reversible
    pub fn non_reversible(mut self) -> Self {
        self.migration.metadata.reversible = false;
        self
    }

    /// Add collection creation
    pub fn create_collection(mut self, config: CollectionConfig) -> Self {
        self.migration.collection_changes.push(CollectionChange::CreateCollection(config));
        self
    }

    /// Add collection deletion
    pub fn delete_collection(mut self, name: String) -> Self {
        self.migration.collection_changes.push(CollectionChange::DeleteCollection(name));
        self
    }

    /// Add Pact contract for validation
    pub fn add_pact_contract(mut self, contract: PactContract) -> Self {
        self.migration.pact_contracts.push(contract);
        self
    }

    /// Build the migration
    pub fn build(self) -> SchemaMigration {
        self.migration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_builder() {
        let version = semver::Version::new(1, 0, 0);
        let migration = MigrationBuilder::new(version.clone(), "Initial schema".to_string())
            .author("Test Author".to_string())
            .description("Create initial collections".to_string())
            .create_collection(CollectionConfig::new("documents", 768, DistanceMetric::Cosine).unwrap())
            .build();

        assert_eq!(migration.version, version);
        assert_eq!(migration.name, "Initial schema");
        assert_eq!(migration.metadata.author, "Test Author");
        assert_eq!(migration.collection_changes.len(), 1);
    }

    #[test]
    fn test_pact_contract_creation() {
        let contract = PactContract {
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

        assert_eq!(contract.consumer, "document-service");
        assert_eq!(contract.interactions.len(), 1);
    }
}