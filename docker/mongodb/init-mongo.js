// MongoDB initialization script
db = db.getSiblingDB('qortfolio');

// Create collections
db.createCollection('options_data');
db.createCollection('price_data');
db.createCollection('portfolio_data');
db.createCollection('risk_metrics');
db.createCollection('user_settings');

// Create indexes for better performance
db.options_data.createIndex({ "symbol": 1, "timestamp": -1 });
db.options_data.createIndex({ "expiry": 1 });
db.price_data.createIndex({ "symbol": 1, "timestamp": -1 });
db.portfolio_data.createIndex({ "user_id": 1, "timestamp": -1 });

print("Qortfolio database initialized successfully!");
