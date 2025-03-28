import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import json
import os

class SemanticEngine:
    def __init__(self, icons_path=None):
        self.icons = []
        # Load predefined Singapore-specific icons
        self.icons = self.load_icons(icons_path)
        
        # Simple embedding simulation (real implementation would use sentence-transformers)
        self.icon_embeddings = self.generate_embeddings()
        
        # Context tracking
        self.recent_selections = []
        self.time_contexts = {
            "morning": [6, 11],
            "afternoon": [12, 17],
            "evening": [18, 21],
            "night": [22, 5]
        }
    
    def load_icons(self, icons_path):
        """Load icon data from file or use defaults"""
        default_icons = [
            {"id": 1, "label": "teh", "category": "drink", "synonyms": ["tea", "drink", "beverage"]},
            {"id": 2, "label": "kopi", "category": "drink", "synonyms": ["coffee", "caffeine"]},
            {"id": 3, "label": "toilet", "category": "need", "synonyms": ["bathroom", "restroom"]},
            {"id": 4, "label": "pain", "category": "need", "synonyms": ["hurt", "ache", "discomfort"]},
            {"id": 5, "label": "nurse", "category": "people", "synonyms": ["help", "medical", "assistance"]},
            {"id": 6, "label": "doctor", "category": "people", "synonyms": ["physician", "medical"]},
            {"id": 7, "label": "too hot", "category": "comfort", "synonyms": ["warm", "heat", "temperature"]},
            {"id": 8, "label": "too cold", "category": "comfort", "synonyms": ["chilly", "cool", "temperature"]},
            {"id": 9, "label": "medicine", "category": "need", "synonyms": ["medication", "drugs", "treatment"]},
            {"id": 10, "label": "family", "category": "people", "synonyms": ["visitors", "relatives"]},
            {"id": 11, "label": "food", "category": "need", "synonyms": ["meal", "eat", "hungry"]},
            {"id": 12, "label": "water", "category": "drink", "synonyms": ["thirsty", "hydrate"]},
            {"id": 13, "label": "TV", "category": "entertainment", "synonyms": ["television", "watch"]},
            {"id": 14, "label": "blanket", "category": "comfort", "synonyms": ["cover", "sheet", "warm"]},
            {"id": 15, "label": "adjust bed", "category": "comfort", "synonyms": ["position", "move"]}
        ]
        
        if icons_path and os.path.exists(icons_path):
            try:
                with open(icons_path, 'r') as f:
                    return json.load(f)
            except:
                pass
                
        return default_icons
    
    def generate_embeddings(self):
        """Generate simple embeddings for icons (placeholder for sentence-transformers)"""
        # In a real implementation, this would use a proper embedding model
        embeddings = {}
        for icon in self.icons:
            # Create a simple embedding based on the label and synonyms
            # This is a placeholder - real implementation would use sentence-transformers
            text = icon["label"] + " " + " ".join(icon.get("synonyms", []))
            embeddings[icon["id"]] = np.random.randn(20)  # Random 20-dim embedding
        return embeddings
    
    def get_current_context(self):
        """Get current context based on time and recent selections"""
        # Time context
        current_hour = datetime.datetime.now().hour
        time_context = "default"
        for context, time_range in self.time_contexts.items():
            if time_range[0] <= current_hour < time_range[1]:
                time_context = context
                
        # Recent selections context (last 3)
        selection_context = [s["category"] for s in self.recent_selections[-3:]] if self.recent_selections else []
        
        return {
            "time": time_context,
            "recent_categories": selection_context
        }
    
    def cluster_icons(self, context=None):
        """Group icons by contextual relevance"""
        if context is None:
            context = self.get_current_context()
            
        # Apply contextual rules
        ranked_icons = []
        
        # First priority: recent categories
        recent_cats = context.get("recent_categories", [])
        priority_categories = set(recent_cats)
        
        # Second priority: time-appropriate categories
        time_context = context.get("time", "default")
        if time_context == "morning":
            priority_categories.add("food")  # Breakfast time
        elif time_context == "afternoon":
            priority_categories.add("entertainment")  # Midday activities
        elif time_context == "evening":
            priority_categories.add("people")  # Family visits
        elif time_context == "night":
            priority_categories.add("comfort")  # Sleep comfort
            
        # Rank icons by priority categories
        for icon in self.icons:
            if icon["category"] in priority_categories:
                ranked_icons.append((icon, 1.0))
            else:
                ranked_icons.append((icon, 0.5))
                
        # Sort by score
        ranked_icons.sort(key=lambda x: -x[1])
        
        # Return top 9 for 3x3 grid
        return ranked_icons[:9]
    
    def update_context(self, selected_icon):
        """Update context based on selection"""
        if selected_icon:
            # Add to recent selections
            self.recent_selections.append(selected_icon)
            # Keep only last 5
            if len(self.recent_selections) > 5:
                self.recent_selections.pop(0)
                
    def get_icon_by_id(self, icon_id):
        """Get icon by ID"""
        for icon in self.icons:
            if icon["id"] == icon_id:
                return icon
        return None
