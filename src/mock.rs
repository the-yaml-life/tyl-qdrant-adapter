//! Mock implementation for testing

use super::*;
use std::sync::{Arc, Mutex};

/// Mock Qdrant adapter for testing
#[derive(Debug, Clone)]
pub struct MockQdrantAdapter {
    collections: Arc<Mutex<HashMap<String, CollectionConfig>>>,
    vectors: Arc<Mutex<HashMap<String, HashMap<String, Vector>>>>, // collection -> id -> vector
}

impl MockQdrantAdapter {
    /// Create a new mock adapter
    pub fn new() -> Self {
        Self {
            collections: Arc::new(Mutex::new(HashMap::new())),
            vectors: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create mock adapter with custom config (for compatibility)
    pub fn with_config(_config: QdrantConfig) -> Self {
        Self::new()
    }
}

impl Default for MockQdrantAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VectorStore for MockQdrantAdapter {
    async fn store_vector(&self, collection: &str, vector: Vector) -> TylResult<()> {
        let mut vectors = self.vectors.lock().unwrap();
        let collection_vectors = vectors.entry(collection.to_string()).or_default();
        collection_vectors.insert(vector.id.clone(), vector);
        Ok(())
    }

    async fn store_vectors_batch(&self, collection: &str, vectors: Vec<Vector>) -> TylResult<Vec<TylResult<()>>> {
        let mut results = Vec::new();
        for vector in vectors {
            let result = self.store_vector(collection, vector).await;
            results.push(result);
        }
        Ok(results)
    }

    async fn get_vector(&self, collection: &str, id: &str) -> TylResult<Option<Vector>> {
        let vectors = self.vectors.lock().unwrap();
        if let Some(collection_vectors) = vectors.get(collection) {
            Ok(collection_vectors.get(id).cloned())
        } else {
            Err(vector_errors::collection_not_found(collection))
        }
    }

    async fn search_similar(&self, collection: &str, _query_vector: Vec<f32>, params: SearchParams) -> TylResult<Vec<VectorSearchResult>> {
        let vectors = self.vectors.lock().unwrap();
        if let Some(collection_vectors) = vectors.get(collection) {
            let mut results = Vec::new();
            for vector in collection_vectors.values() {
                // Simple mock: return vectors that match filters
                let matches_filter = if params.filters.is_empty() {
                    true
                } else {
                    params.filters.iter().all(|(key, value)| {
                        vector.metadata.get(key) == Some(value)
                    })
                };

                if matches_filter {
                    let result = VectorSearchResult::new(vector.clone(), 0.9); // Mock score
                    results.push(result);
                }

                if results.len() >= params.limit {
                    break;
                }
            }
            Ok(results)
        } else {
            Err(vector_errors::collection_not_found(collection))
        }
    }

    async fn delete_vector(&self, collection: &str, id: &str) -> TylResult<()> {
        let mut vectors = self.vectors.lock().unwrap();
        if let Some(collection_vectors) = vectors.get_mut(collection) {
            collection_vectors.remove(id);
            Ok(())
        } else {
            Err(vector_errors::collection_not_found(collection))
        }
    }

    async fn delete_vectors_batch(&self, collection: &str, ids: Vec<String>) -> TylResult<()> {
        for id in ids {
            self.delete_vector(collection, &id).await?;
        }
        Ok(())
    }
}

#[async_trait]
impl VectorCollectionManager for MockQdrantAdapter {
    async fn create_collection(&self, config: CollectionConfig) -> TylResult<()> {
        let mut collections = self.collections.lock().unwrap();
        if collections.contains_key(&config.name) {
            return Err(vector_errors::storage_failed(format!("Collection '{}' already exists", config.name)));
        }
        let collection_name = config.name.clone();
        collections.insert(collection_name.clone(), config);
        
        // Initialize empty vector storage for this collection
        let mut vectors = self.vectors.lock().unwrap();
        vectors.insert(collection_name, HashMap::new());
        
        Ok(())
    }

    async fn delete_collection(&self, collection_name: &str) -> TylResult<()> {
        let mut collections = self.collections.lock().unwrap();
        let mut vectors = self.vectors.lock().unwrap();
        
        collections.remove(collection_name);
        vectors.remove(collection_name);
        
        Ok(())
    }

    async fn list_collections(&self) -> TylResult<Vec<CollectionConfig>> {
        let collections = self.collections.lock().unwrap();
        Ok(collections.values().cloned().collect())
    }

    async fn get_collection_info(&self, collection_name: &str) -> TylResult<Option<CollectionConfig>> {
        let collections = self.collections.lock().unwrap();
        Ok(collections.get(collection_name).cloned())
    }

    async fn get_collection_stats(&self, collection_name: &str) -> TylResult<HashMap<String, serde_json::Value>> {
        let vectors = self.vectors.lock().unwrap();
        let mut stats = HashMap::new();
        
        if let Some(collection_vectors) = vectors.get(collection_name) {
            stats.insert("vectors_count".to_string(), serde_json::json!(collection_vectors.len()));
            stats.insert("status".to_string(), serde_json::json!("green"));
        } else {
            return Err(vector_errors::collection_not_found(collection_name));
        }
        
        Ok(stats)
    }
}

#[async_trait]
impl VectorStoreHealth for MockQdrantAdapter {
    async fn is_healthy(&self) -> TylResult<bool> {
        Ok(true)
    }

    async fn health_check(&self) -> TylResult<HashMap<String, serde_json::Value>> {
        let mut health = HashMap::new();
        health.insert("status".to_string(), serde_json::json!("healthy"));
        health.insert("type".to_string(), serde_json::json!("mock"));
        Ok(health)
    }
}

#[async_trait]
impl VectorDatabase for MockQdrantAdapter {
    type Config = QdrantConfig;

    async fn connect(config: Self::Config) -> VectorResult<Self> {
        Ok(Self::with_config(config))
    }

    fn connection_info(&self) -> String {
        "Mock Qdrant Adapter (for testing)".to_string()
    }

    async fn close(&mut self) -> VectorResult<()> {
        Ok(())
    }

    fn supports_feature(&self, feature: &str) -> bool {
        matches!(feature, "collections" | "health_check" | "batch_operations" | "filtering")
    }
}