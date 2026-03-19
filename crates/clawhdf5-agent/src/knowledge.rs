//! Knowledge graph data structures and cache.

/// A knowledge graph entity.
#[derive(Debug, Clone)]
pub struct Entity {
    pub id: u64,
    pub name: String,
    pub entity_type: String,
    /// Index into the memory embeddings array, or -1 if none.
    pub embedding_idx: i64,
}

/// A knowledge graph relation between two entities.
#[derive(Debug, Clone)]
pub struct Relation {
    pub src: u64,
    pub tgt: u64,
    pub relation: String,
    pub weight: f32,
    pub ts: f64,
}

/// In-memory cache for the /knowledge_graph group.
#[derive(Debug, Clone)]
pub struct KnowledgeCache {
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
    pub alias_strings: Vec<String>,
    pub alias_entity_ids: Vec<i64>,
    next_entity_id: u64,
}

impl KnowledgeCache {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            relations: Vec::new(),
            alias_strings: Vec::new(),
            alias_entity_ids: Vec::new(),
            next_entity_id: 0,
        }
    }

    pub fn new_with_next_id(next_id: u64) -> Self {
        Self {
            entities: Vec::new(),
            relations: Vec::new(),
            alias_strings: Vec::new(),
            alias_entity_ids: Vec::new(),
            next_entity_id: next_id,
        }
    }

    /// Add an entity, returns its assigned ID.
    pub fn add_entity(&mut self, name: &str, entity_type: &str, embedding_idx: i64) -> u64 {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        self.entities.push(Entity {
            id,
            name: name.to_string(),
            entity_type: entity_type.to_string(),
            embedding_idx,
        });
        id
    }

    /// Add a relation between two entities.
    pub fn add_relation(&mut self, src: u64, tgt: u64, relation: &str, weight: f32) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
            * 1_000_000.0;
        self.relations.push(Relation {
            src,
            tgt,
            relation: relation.to_string(),
            weight,
            ts,
        });
    }

    /// Find an entity by ID.
    pub fn get_entity(&self, id: u64) -> Option<&Entity> {
        self.entities.iter().find(|e| e.id == id)
    }

    /// Find all relations where the given entity is the source.
    pub fn get_relations_from(&self, src_id: u64) -> Vec<&Relation> {
        self.relations.iter().filter(|r| r.src == src_id).collect()
    }

    /// Find all relations where the given entity is the target.
    pub fn get_relations_to(&self, tgt_id: u64) -> Vec<&Relation> {
        self.relations.iter().filter(|r| r.tgt == tgt_id).collect()
    }

    /// Register an alias for an entity. Case-insensitive storage.
    pub fn add_alias(&mut self, alias: &str, entity_id: i64) {
        self.alias_strings.push(alias.to_lowercase());
        self.alias_entity_ids.push(entity_id);
    }

    /// Get all aliases for a given entity.
    pub fn get_aliases(&self, entity_id: i64) -> Vec<&str> {
        self.alias_strings
            .iter()
            .zip(&self.alias_entity_ids)
            .filter(|&(_, id)| *id == entity_id)
            .map(|(s, _)| s.as_str())
            .collect()
    }

    /// Get entity name by ID.
    pub fn get_entity_name(&self, entity_id: i64) -> Option<&str> {
        self.entities
            .iter()
            .find(|e| e.id == entity_id as u64)
            .map(|e| e.name.as_str())
    }

    /// Resolve aliases in free text — greedy longest-match replacement.
    pub fn resolve_aliases(&self, query: &str) -> String {
        let lower = query.to_lowercase();
        let mut pairs: Vec<(&str, String)> = self
            .alias_strings
            .iter()
            .zip(&self.alias_entity_ids)
            .filter_map(|(alias, &eid)| {
                self.get_entity_name(eid)
                    .map(|name| (alias.as_str(), name.to_lowercase()))
            })
            .collect();
        pairs.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let mut result = lower;
        for (alias, name) in &pairs {
            result = result.replace(alias, name);
        }
        result
    }
}

impl Default for KnowledgeCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get_aliases() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Henry", "person", -1);
        cache.add_alias("my son", id as i64);
        cache.add_alias("the kid", id as i64);

        let aliases = cache.get_aliases(id as i64);
        assert_eq!(aliases.len(), 2);
        assert!(aliases.contains(&"my son"));
        assert!(aliases.contains(&"the kid"));
    }

    #[test]
    fn test_resolve_single_alias() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Henry", "person", -1);
        cache.add_alias("my son", id as i64);

        let resolved = cache.resolve_aliases("what does my son do?");
        assert!(resolved.contains("henry"), "expected 'henry' in '{resolved}'");
    }

    #[test]
    fn test_resolve_multiple_aliases() {
        let mut cache = KnowledgeCache::new();
        let h = cache.add_entity("Henry", "person", -1);
        let a = cache.add_entity("Acme Corp", "company", -1);
        cache.add_alias("my son", h as i64);
        cache.add_alias("our main client", a as i64);

        let resolved = cache.resolve_aliases("what does my son do at our main client?");
        assert!(resolved.contains("henry"), "expected 'henry' in '{resolved}'");
        assert!(resolved.contains("acme corp"), "expected 'acme corp' in '{resolved}'");
    }

    #[test]
    fn test_longest_match_wins() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Henry", "person", -1);
        cache.add_alias("my son", id as i64);
        cache.add_alias("my son henry", id as i64);

        let resolved = cache.resolve_aliases("ask my son henry");
        // The longer alias "my son henry" should match first, producing "ask henry"
        // If "my son" matched first, we'd get "ask henry henry" — wrong.
        assert_eq!(resolved, "ask henry");
    }

    #[test]
    fn test_unregistered_alias_passthrough() {
        let mut cache = KnowledgeCache::new();
        cache.add_entity("Henry", "person", -1);
        // No aliases registered
        let resolved = cache.resolve_aliases("unknown phrase here");
        assert_eq!(resolved, "unknown phrase here");
    }

    #[test]
    fn test_case_insensitive_resolve() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Henry", "person", -1);
        cache.add_alias("henry", id as i64);

        let resolved = cache.resolve_aliases("Tell HENRY about it");
        assert!(resolved.contains("henry"), "expected 'henry' in '{resolved}'");
    }

    #[test]
    fn test_empty_aliases() {
        let cache = KnowledgeCache::new();
        let resolved = cache.resolve_aliases("hello");
        assert_eq!(resolved, "hello");
    }

    #[test]
    fn test_get_entity_name() {
        let mut cache = KnowledgeCache::new();
        let id = cache.add_entity("Henry", "person", -1);
        assert_eq!(cache.get_entity_name(id as i64), Some("Henry"));
        assert_eq!(cache.get_entity_name(999), None);
    }
}
