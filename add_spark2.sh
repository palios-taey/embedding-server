#!/bin/bash
# Add Spark 2 back to load balancer after recovery
# Run this after Spark 2 is rebooted and embedding instances are started

echo "Checking Spark 2 instances..."
for port in 8081 8082 8083 8084; do
    if curl -s --connect-timeout 5 http://10.0.0.80:$port/health > /dev/null 2>&1; then
        echo "  Port $port: healthy"
    else
        echo "  Port $port: not responding"
        echo "ERROR: Spark 2 instance on port $port is not healthy"
        echo "Make sure to start instances on Spark 2 first:"
        echo "  ssh spark@10.0.0.80 'NUM_INSTANCES=4 /home/spark/embedding-server/start-sequential.sh'"
        exit 1
    fi
done

echo ""
echo "All Spark 2 instances healthy. Updating load balancer config..."

cat > /home/spark/embedding-server/nginx-lb.conf << 'EOF'
# NGINX Load Balancer for Qwen3-Embedding-8B
# 4 instances per Spark (8 total)

upstream embedding_servers {
    # Spark 1 instances
    server 10.0.0.68:8081;
    server 10.0.0.68:8082;
    server 10.0.0.68:8083;
    server 10.0.0.68:8084;

    # Spark 2 instances
    server 10.0.0.80:8081;
    server 10.0.0.80:8082;
    server 10.0.0.80:8083;
    server 10.0.0.80:8084;

    keepalive 16;
}

server {
    listen 8090;
    server_name _;

    location /health {
        proxy_pass http://embedding_servers/health;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }

    location /embed {
        proxy_pass http://embedding_servers/embed;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_read_timeout 300s;
        proxy_connect_timeout 10s;
        proxy_send_timeout 300s;
    }

    location / {
        proxy_pass http://embedding_servers/;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
EOF

docker exec embedding-lb nginx -s reload
echo ""
echo "Load balancer updated! Now serving from both Sparks."
echo "Combined throughput: ~5000 tok/s (2 GPUs)"
echo ""
echo "Test: curl http://localhost:8090/health"
