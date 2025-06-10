
## APOC Issue:

```
docker run -d \                                       
    --publish=7474:7474 --publish=7687:7687 \
    --volume=neo4j-data:/data \
    --name neo4j-ctn --env='NEO4JLABS_PLUGINS=["apoc"]' neo4j:5.26.8-community-ubi9
```
