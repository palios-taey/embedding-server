#!/bin/bash
# Launch full ISMA Neo4j reprocessing
cd /home/spark/embedding-server/isma/scripts
rm -f /var/spark/isma/reprocess_progress.json /var/spark/isma/reprocess_hashes.json
exec /usr/bin/python3 reprocess_neo4j.py --all >> /var/spark/isma/reprocess.log 2>&1
