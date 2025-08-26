//! Test to explore qdrant-client API options

use qdrant_client::*;
use std::time::Duration;

#[tokio::main]
async fn main() {
    println!("Testing Qdrant client API...");
    
    // Try different builder configurations
    let builder = Qdrant::from_url("http://localhost:6333")
        .timeout(Duration::from_secs(30));
    
    println!("Checking available methods...");
    
    // Try setting compatibility check through environment or config
    std::env::set_var("QDRANT_CHECK_COMPATIBILITY", "false");
    
    // Try the builder with different config approaches
    println!("Attempting to configure compatibility checking...");
    
    // Let's see what config options are available
    println!("Exploring config options...");
    
    // Try to find compatibility/version check methods
    println!("Builder created successfully");
    
    // What happens if we try to build?
    match builder.build() {
        Ok(_client) => println!("Client built successfully!"),
        Err(e) => println!("Client build failed: {}", e),
    }
}