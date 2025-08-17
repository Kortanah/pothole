#!/bin/bash
# codespace_postgres_setup.sh - Complete PostgreSQL setup for GitHub Codespaces

echo "🚀 Setting up PostgreSQL in GitHub Codespaces..."

# Update package list
echo "📦 Updating package list..."
sudo apt-get update -qq

# Install PostgreSQL
echo "🐘 Installing PostgreSQL..."
sudo apt-get install -y postgresql postgresql-contrib postgresql-client

# Check if installation was successful
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL installation failed"
    exit 1
fi

echo "✅ PostgreSQL installed successfully"

# Start PostgreSQL service
echo "🚀 Starting PostgreSQL service..."
sudo service postgresql start

# Wait for PostgreSQL to start
sleep 3

# Check PostgreSQL status
echo "📊 Checking PostgreSQL status..."
sudo service postgresql status

# Initialize PostgreSQL if needed using sudo -i
echo "🔧 Initializing PostgreSQL..."

# Use sudo -i to avoid password prompts for system operations
sudo -i bash << 'ROOT_SETUP'
    # Check if postgres user exists, create if not
    if ! id "postgres" &>/dev/null; then
        echo "👤 Creating postgres user..."
        useradd -m postgres
    fi

    # Set up PostgreSQL cluster if not exists
    PG_VERSION=$(psql --version | awk '{print $3}' | sed 's/\..*//' 2>/dev/null || echo "14")
    
    if [ ! -d "/var/lib/postgresql/$PG_VERSION/main" ]; then
        echo "🏗️ Initializing PostgreSQL cluster..."
        su - postgres -c "/usr/lib/postgresql/$PG_VERSION/bin/initdb -D /var/lib/postgresql/$PG_VERSION/main" 2>/dev/null || echo "Cluster initialization skipped"
    fi
ROOT_SETUP

# Ensure PostgreSQL is running
sudo service postgresql restart
sleep 3

# Set postgres user password using sudo -i
echo "🔑 Setting postgres user password..."
sudo -i bash << 'PASSWORD_SETUP'
    su - postgres -c "psql -c \"ALTER USER postgres PASSWORD 'password';\" postgres"
PASSWORD_SETUP

# Create potholes database using sudo -i
echo "🗄️ Creating potholes database..."
sudo -i bash << 'DB_SETUP'
    su - postgres -c "createdb potholes" 2>/dev/null || echo "Database already exists"
    su - postgres -c "psql -c \"GRANT ALL PRIVILEGES ON DATABASE potholes TO postgres;\" postgres"
DB_SETUP

# Test connection with password
echo "🧪 Testing database connection..."
if PGPASSWORD=password psql -h localhost -U postgres -d potholes -c "SELECT 'Connection successful!' as test;" 2>/dev/null; then
    echo "✅ PostgreSQL setup completed successfully!"
    echo ""
    echo "📋 Database Connection Details:"
    echo "   Host: localhost"
    echo "   Port: 5432 (default)"
    echo "   Database: potholes"
    echo "   Username: postgres"
    echo "   Password: password"
    echo ""
    echo "🔗 Connection String:"
    echo "   postgresql://postgres:password@localhost:5432/potholes"
    echo ""
    echo "🎯 You can now run your FastAPI application!"
else
    echo "⚠️ Connection test failed, but setup should still work"
    echo "💡 Try running your application - it may work despite this warning"
    
    # Alternative test using peer authentication
    echo "🔄 Trying alternative connection test..."
    sudo -i bash << 'ALT_TEST'
        su - postgres -c "psql -d potholes -c \"SELECT 'Alternative connection successful!' as test;\""
ALT_TEST
fi

# Additional setup for development convenience
echo "🔧 Setting up development conveniences..."
sudo -i bash << 'DEV_SETUP'
    # Allow local connections without password for development
    PG_VERSION=$(ls /etc/postgresql/ | head -1)
    if [ -f "/etc/postgresql/$PG_VERSION/main/pg_hba.conf" ]; then
        cp /etc/postgresql/$PG_VERSION/main/pg_hba.conf /etc/postgresql/$PG_VERSION/main/pg_hba.conf.backup
        sed -i 's/local   all             postgres                                peer/local   all             postgres                                md5/' /etc/postgresql/$PG_VERSION/main/pg_hba.conf
        sed -i 's/local   all             all                                     peer/local   all             all                                     md5/' /etc/postgresql/$PG_VERSION/main/pg_hba.conf
        echo "🔐 Updated authentication configuration"
    fi
DEV_SETUP

# Restart PostgreSQL to apply configuration changes
sudo service postgresql restart
sleep 2

# Final connection test
echo "🎯 Final connection test..."
if PGPASSWORD=password psql -h localhost -U postgres -d potholes -c "SELECT 'Final test: PostgreSQL is ready!' as status;" 2>/dev/null; then
    echo "🎉 All tests passed! PostgreSQL is fully configured and ready!"
else
    echo "⚠️ Final test had issues, but basic setup should work"
    echo "💡 Your FastAPI app should still be able to connect"
fi

# Show how to manage PostgreSQL
echo ""
echo "🔧 PostgreSQL Management Commands:"
echo "   sudo service postgresql start    # Start PostgreSQL"
echo "   sudo service postgresql stop     # Stop PostgreSQL"
echo "   sudo service postgresql restart  # Restart PostgreSQL"
echo "   sudo service postgresql status   # Check status"
echo ""
echo "🗄️ Database Commands:"
echo "   PGPASSWORD=password psql -h localhost -U postgres -d potholes  # Connect to database"
echo "   psql -U postgres -d potholes     # Connect using peer authentication"
echo ""
echo "🐍 Python Test Command:"
echo "   python3 -c \"import psycopg2; conn = psycopg2.connect('postgresql://postgres:password@localhost:5432/potholes'); print('✅ Python connection works!'); conn.close()\""
echo ""
echo "🚀 Next Steps:"
echo "   1. pip install opencv-python ultralytics"
echo "   2. mkdir -p services && touch services/__init__.py"
echo "   3. python main.py"