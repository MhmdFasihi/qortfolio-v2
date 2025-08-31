// Switch to qortfolio database
db = db.getSiblingDB('qortfolio');

// Create indexes for better performance
db.price_data.createIndex({ "symbol": 1, "timestamp": -1 });
db.price_data.createIndex({ "timestamp": -1 });

db.options_data.createIndex({ "symbol": 1, "expiry": 1 });
db.options_data.createIndex({ "underlying": 1, "timestamp": -1 });
db.options_data.createIndex({ "strike": 1, "expiry": 1 });

db.portfolio_data.createIndex({ "user_id": 1, "timestamp": -1 });
db.risk_metrics.createIndex({ "portfolio_id": 1, "timestamp": -1 });

print("✅ Indexes created successfully");

// Show collection stats
db.getCollectionNames().forEach(function(collection) {
    var count = db[collection].count();
    print(collection + ": " + count + " documents");
});
