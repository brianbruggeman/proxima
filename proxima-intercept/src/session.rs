use std::collections::HashMap;
use std::sync::Mutex;

pub struct SessionCache {
    summaries: Mutex<HashMap<String, String>>,
}

impl Default for SessionCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionCache {
    pub fn new() -> Self {
        Self {
            summaries: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_summary(&self, session_id: &str) -> Option<String> {
        self.summaries.lock().ok()?.get(session_id).cloned()
    }

    pub fn store_summary(&self, session_id: String, summary: String) {
        if let Ok(mut map) = self.summaries.lock() {
            map.insert(session_id, summary);
        }
    }
}
